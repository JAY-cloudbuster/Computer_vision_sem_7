"""Verify VGG16 deliverables and ensure pre-existing teammate work is preserved."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from modules.vgg16_training import classification_metrics, write_json
from modules.vgg16_evaluation import summarize_reader, text_metrics


def main():
    allowed={"results/cnn_comparison_results.csv","results/reader_comparison_results.csv"}
    changed=set(subprocess.check_output(["git","diff","HEAD","--name-only"],text=True).splitlines())
    assert changed <= allowed, f"Unexpected changes to original files: {changed-allowed}"
    for path in allowed:
        original=pd.read_csv(io.StringIO(subprocess.check_output(["git","show",f"HEAD:{path}"],text=True)))
        current=pd.read_csv(path)
        pd.testing.assert_frame_equal(original[original.Architecture!="VGG16"].reset_index(drop=True),
                                      current[current.Architecture!="VGG16"].reset_index(drop=True),check_dtype=False)
        assert len(current[current.Architecture=="VGG16"])==1
    training=json.loads(Path("results/vgg16/training_details.json").read_text())
    predictions=pd.read_csv("results/vgg16/test_predictions.csv")
    recomputed=classification_metrics(predictions.label,predictions.plate_probability)
    for key in ["Accuracy","Precision","Recall","F1-Score"]:
        assert abs(recomputed[key]-training['test_metrics'][key])<1e-9
    assert recomputed['confusion_matrix']==training['test_metrics']['confusion_matrix']
    assert recomputed['Recall']>0 and recomputed['Precision']>0
    assert set(predictions.prediction)=={0,1}, "Model predicts only one class"
    frame=pd.read_csv("results/vgg16_reader/reader_predictions.csv",keep_default_na=False)
    labels=pd.read_csv("combined_dataset/labels.csv",keep_default_na=False)
    assert len(frame)==len(labels)==1696, "Full reader evaluation is incomplete"
    assert frame.filename.tolist()==labels.filename.tolist()
    for _,r in frame.iterrows():
        assert (r.plate_probability>=.5)==bool(r.accepted)
        assert r.accepted or (r.prediction=="" and r.ocr_ms==0)
        assert np.isclose(r.cnn_ms+r.ocr_ms,r.total_ms)
        metrics=text_metrics(r.prediction,r.gt_text)
        for key in ['exact_match','edit_distance','ned','char_accuracy']:
            assert np.isclose(r[key],metrics[key])
    reader_metrics=json.loads(Path("results/vgg16_reader/reader_metrics.json").read_text())
    assert summarize_reader(frame)==reader_metrics['full_dataset']
    assert summarize_reader(frame.iloc[:40])==reader_metrics['first_40']
    expected=[*Path("results/vgg16").glob("vgg16_sample_case_*.jpg"),
              *Path("results/vgg16_reader").glob("case_*.jpg")]
    assert len(expected)==10
    expected += [Path("results/vgg16/vgg16_all_5_cases_overview.jpg"),
                 Path("results/vgg16_reader/vgg16_all_5_reader_cases_overview.jpg"),
                 Path("results/vgg16/training_and_test_results.png"),Path("results/vgg16/reader_results.png")]
    dimensions={}
    for path in expected:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.width>=1000 and image.height>=500
            dimensions[str(path)]=list(image.size)
    for path in ["models/vgg16_plate_classifier.pth","results/vgg16/VGG16_REPORT.md","VGG16_RUN_GUIDE.md","requirements-vgg16-lock.txt"]:
        assert Path(path).stat().st_size>0
    assert Path("models/vgg16_plate_classifier.pth").stat().st_size>500_000_000
    verification={"status":"passed", "original_files_changed":sorted(changed),
                  "teammate_rows_preserved":True,"reader_records_verified":len(frame),
                  "classification_records_verified":len(predictions),"figure_dimensions":dimensions}
    write_json("results/vgg16/verification.json",verification)
    print(json.dumps(verification,indent=2))


if __name__=="__main__":
    main()
