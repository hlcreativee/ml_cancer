from flask import Flask, render_template, request
import numpy as np
import pickle
import subprocess
from sklearn.tree import export_graphviz, _tree

app = Flask(__name__)

# =============================
# LOAD MODEL BREAST CANCER
# =============================
model_id3 = pickle.load(open("cancer_id3.pkl", "rb"))
model_nb = pickle.load(open("cancer_nb.pkl", "rb"))

# =============================
# LIST FITUR CANCER (30 FEATURES)
# =============================
FEATURES_CANCER = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# =============================
# HOME
# =============================
@app.route("/")
def home():
    return render_template("index.html", fields=FEATURES_CANCER)

# =============================
# PREDIKSI ID3
# =============================
@app.route("/predict_id3", methods=["POST"])
def predict_id3():
    features = [float(request.form[f]) for f in FEATURES_CANCER]
    pred = model_id3.predict([features])[0]
    return render_template("index.html", hasil_id3=f"Hasil ID3: {pred}", fields=FEATURES_CANCER)

# =============================
# PREDIKSI NAIVE BAYES
# =============================
@app.route("/predict_nb", methods=["POST"])
def predict_nb():
    features = [float(request.form[f]) for f in FEATURES_CANCER]
    pred = model_nb.predict([features])[0]
    return render_template("index.html", hasil_nb=f"Hasil Naive Bayes: {pred}", fields=FEATURES_CANCER)

# =============================
# HELPER: EXTRACT RULES DARI ID3
# =============================
def extract_rules(model, feature_names):
    tree = model.tree_
    rules = []

    def traverse(node, rule_text):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            # kiri: <=
            left_rule = rule_text + [f"{name} <= {threshold:.3f}"]
            traverse(tree.children_left[node], left_rule)

            # kanan: >
            right_rule = rule_text + [f"{name} > {threshold:.3f}"]
            traverse(tree.children_right[node], right_rule)
        else:
            class_value = model.classes_[tree.value[node].argmax()]
            rule_sentence = "JIKA " + " DAN ".join(rule_text) + f" → MAKA kelas = {class_value}"
            rules.append(rule_sentence)

    traverse(0, [])
    return rules

# =============================
# HALAMAN POHON + RULES
# =============================
@app.route("/tree")
def tree():
    # Export pohon ke .dot
    dot_file = "tree_temp.dot"
    export_graphviz(
        model_id3,
        out_file=dot_file,
        feature_names=FEATURES_CANCER,
        class_names=["B", "M"],
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True
    )

    # Convert .dot ke SVG
    svg_bytes = subprocess.check_output(["dot", "-Tsvg", dot_file])
    svg_text = svg_bytes.decode("utf-8")

    # Extract rules
    rules_list = extract_rules(model_id3, FEATURES_CANCER)

    # Render halaman tree + rules
    return render_template("tree.html", tree_image=svg_text, rules=rules_list)

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    app.run(debug=True)
