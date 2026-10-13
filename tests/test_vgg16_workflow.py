"""Regression checks for VGG16-specific correctness improvements."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from modules.vgg16_evaluation import clean_prediction, normalize_text, text_metrics, summarize_reader, align_characters
from modules.vgg16_training import update_only_vgg_row, classification_metrics


@pytest.mark.parametrize("text", ["KA19TR02", "MH20CS9817", "KL05AK3300", "MH12JC2813", "UP50AS4535"])
def test_valid_plate_is_preserved(text):
    assert clean_prediction(text) == text
    assert normalize_text(text) == text
    assert text_metrics(text, text)["exact_match"] == 1


def test_ground_truth_is_not_repaired_to_match_wrong_prediction():
    assert text_metrics("KA191R02", "KA19TR02")["exact_match"] == 0
    assert text_metrics("MH20CS9817", "MH20CS9817")["ned"] == 1
    assert text_metrics("", "KA19TR02")["char_accuracy"] == 0
    assert normalize_text("AB123456789012") == "AB123456789012"


def test_edit_distance_accounts_for_insertions():
    result = text_metrics("ABX123", "AB123")
    assert result["edit_distance"] == 1
    assert result["char_accuracy"] == pytest.approx(.8)


@pytest.mark.parametrize("truth,prediction",[("AB123","ABX123"),("ABX123","AB123")])
def test_character_alignment_does_not_mark_the_entire_suffix_wrong(truth,prediction):
    pairs=align_characters(truth,prediction)
    assert sum(a!=b for a,b in pairs)==1
    assert ''.join(a for a,b in pairs if a!='-')==truth
    assert ''.join(b for a,b in pairs if b!='-')==prediction
    assert pairs[-3:]==[("1","1"),("2","2"),("3","3")]


def test_vgg_update_preserves_teammates_and_rejects_malformed_table(tmp_path):
    file=tmp_path/"results.csv"
    pd.DataFrame([{"Architecture":"ResNet50","Accuracy":.946},
                  {"Architecture":"VGG16","Accuracy":.6667}]).to_csv(file,index=False)
    update_only_vgg_row(file,{"Architecture":"VGG16","Accuracy":.97})
    result=pd.read_csv(file)
    assert result[result.Architecture=="ResNet50"].Accuracy.iloc[0]==.946
    assert len(result[result.Architecture=="VGG16"])==1
    file.write_text("bad,columns\n1,2\n")
    before=file.read_bytes()
    with pytest.raises(ValueError):
        update_only_vgg_row(file,{"Architecture":"VGG16","Accuracy":.97})
    assert file.read_bytes()==before


def test_rejected_plate_remains_in_reader_denominator_and_timing():
    frame=pd.DataFrame([dict(accepted=1,exact_match=1,ned=1.,char_accuracy=1.,cnn_ms=10.,ocr_ms=100.,total_ms=110.),
                        dict(accepted=0,exact_match=0,ned=0.,char_accuracy=0.,cnn_ms=10.,ocr_ms=0.,total_ms=10.)])
    summary=summarize_reader(frame)
    assert summary["Exact Match (%)"]==50
    assert summary["End-to-End Latency (ms)"]==60
    assert summary["Verified by Gate"]==1


def test_collapsed_classifier_cannot_hide_behind_accuracy():
    result=classification_metrics([0,0,1],[.1,.1,.1])
    assert result["Accuracy"]==pytest.approx(2/3)
    assert result["Recall"]==0
    assert result["F1-Score"]==0


def test_saved_splits_are_disjoint_when_available():
    path=Path("results/vgg16/crop_manifest.json")
    if not path.exists():
        pytest.skip("Training manifest has not been generated")
    records=json.loads(path.read_text())["records"]
    ids={s:{r['source_group'] for r in rows} for s,rows in records.items()}
    assert not ids['train'] & ids['validation']
    assert not (ids['train'] | ids['validation']) & ids['test']


def test_checkpoint_matches_recorded_predictions_when_available():
    path=Path("models/vgg16_plate_classifier.pth")
    if not path.exists():
        pytest.skip("Training checkpoint has not been generated")
    from modules.case_visualizer import load_trained_model
    from modules.vgg16_training import Crops
    torch.set_num_threads(4)
    model=load_trained_model("VGG16",str(path),"cpu")
    manifest=json.loads(Path("results/vgg16/crop_manifest.json").read_text())
    rows=pd.read_csv("results/vgg16/test_predictions.csv")
    data=Crops(manifest['records']['test'])
    # Include both labels instead of testing only one easy example.
    indices=[int(rows[rows.label==label].index[0]) for label in [0,1]]
    with torch.inference_mode():
        for index in indices:
            x,_=data[index]
            logits=model(x.unsqueeze(0))
            p=float(logits.softmax(1)[0,1])
            assert p==pytest.approx(rows.iloc[index].plate_probability,abs=2e-5)
            features=model.classifier[:6](model.avgpool(model.features(x.unsqueeze(0))).flatten(1))
            assert torch.allclose(model.classifier[6](features),logits,atol=1e-6)
