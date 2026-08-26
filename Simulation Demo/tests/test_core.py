from random import Random

from app.classifier import predict, train
from app.physics import evaluate_pit, simulate_press
from app.synthetic_data import make_record


def test_edema_retains_dent_and_is_detected() -> None:
    normal = simulate_press("normal", force=7, duration=3)
    edema = simulate_press("edema", force=7, duration=3)
    assert edema["peak_depth"] > normal["peak_depth"] * 0.9
    assert edema["residual_depth"] > 0
    assert normal["residual_depth"] == 0
    assert not evaluate_pit(normal)["pit_detected"]
    assert evaluate_pit(edema)["pit_detected"]


def test_synthetic_labels_share_physics_and_are_varied() -> None:
    rng = Random(21)
    yes = make_record(True, rng)
    no = make_record(False, rng)
    another_yes = make_record(True, rng)
    assert yes["label"]["tissue_mode"] == "edema"
    assert no["label"]["tissue_mode"] == "normal"
    assert yes["press"]["summary"]["deterministic_pit"]["pit_detected"]
    assert not no["press"]["summary"]["deterministic_pit"]["pit_detected"]
    assert yes["blood_values"]["albumin_g_dL"] < 3.0
    assert no["blood_values"]["albumin_g_dL"] >= 3.5
    assert yes["blood_values"]["hemoglobin_g_dL"] != another_yes["blood_values"]["hemoglobin_g_dL"]


def test_patient_record_respects_selected_press_protocol() -> None:
    record = make_record(True, Random(5), force=11.5, duration=4.0)
    assert record["press"]["force"] == 11.5
    assert record["press"]["duration"] == 4.0


def test_small_visual_classifier_learns_from_rendered_sequences() -> None:
    rng = Random(9)
    records = [make_record(index % 2 == 0, rng) for index in range(30)]
    model = train(records, epochs=240)
    yes_prediction = predict(model, records[0]["press"]["time_series"], records[0]["press"]["duration"])
    no_prediction = predict(model, records[1]["press"]["time_series"], records[1]["press"]["duration"])
    assert model["training_accuracy"] >= 0.8
    assert yes_prediction["pit_detected"]
    assert not no_prediction["pit_detected"]
