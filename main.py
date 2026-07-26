import argparse
import csv
import glob
import json
import os

import yaml
from flask import Flask, redirect, render_template, send_from_directory
from flask_frozen import Freezer
from flaskext.markdown import Markdown

site_data = {}
site_data_path = ""
extra_files = []
WATCHED_MARKDOWN = ["Alumni.md", "HKUST.md", "PreHKUST.md"]


def main(site_data_dir):
    """Load configuration data once when the server starts."""
    global site_data, extra_files, site_data_path
    site_data_path = site_data_dir
    site_data = {}
    extra_files = []

    for path in glob.glob(os.path.join(site_data_dir, "*")):
        extra_files.append(path)
        name, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lstrip(".").lower()

        if ext == "json":
            with open(path, encoding="utf-8") as handle:
                site_data[name] = json.load(handle)
        elif ext in {"csv", "tsv"}:
            with open(path, encoding="utf-8") as handle:
                site_data[name] = list(csv.DictReader(handle))
        elif ext in {"yml", "yaml"}:
            with open(path, encoding="utf-8") as handle:
                site_data[name] = yaml.load(handle, Loader=yaml.SafeLoader)

    for watched in WATCHED_MARKDOWN:
        if os.path.exists(watched):
            extra_files.append(watched)

    return extra_files


app = Flask(__name__)
app.config.from_object(__name__)
freezer = Freezer(app)
Markdown(app)


def _data():
    """Base context that every template expects."""
    return {"config": site_data.get("config", {})}


@app.route("/")
def index():
    return redirect("/index.html")


@app.route("/logo.jpg")
def favicon():
    return send_from_directory(site_data_path, "logo.jpg")


@app.route("/index.html")
def home():
    return render_template("index.html", **_data())


@app.route("/overview.html")
def overview():
    return render_template("overview.html", **_data())


@app.route("/simulator.html")
def simulator():
    return render_template("simulator.html", **_data())


@app.route("/emulator.html")
def emulator():
    return render_template("emulator.html", **_data())


@app.route("/sampler.html")
def sampler():
    return render_template("sampler.html", **_data())


@app.route("/predictor.html")
def predictor():
    return render_template("predictor.html", **_data())


@app.route("/evaluator_agent.html")
def evaluator_agent():
    return render_template("evaluator_agent.html", **_data())


@app.route("/members.html")
def members():
    data = _data()
    committee = site_data.get("committee", {}).get("committee", [])
    data["CurrentMember"] = committee
    if os.path.exists("Alumni.md"):
        with open("Alumni.md", encoding="utf-8") as handle:
            data["Alumni"] = handle.read()
    else:
        data["Alumni"] = ""
    return render_template("members.html", **data)


@app.route("/openings.html")
def openings():
    return render_template("openings.html", **_data())


@app.route("/news.html")
def news():
    return render_template("news.html", **_data())


@app.route("/mini_tests.html")
def mini_tests():
    return render_template("mini_tests.html", **_data())


@app.route("/hkust_credits.html")
def hkust_credits():
    return render_template("hkust_credits.html", **_data())


@app.route("/summer_research_2026.html")
def summer_research_2026():
    return render_template("summer_research_2026.html", **_data())


@app.route("/tools.html")
def tools():
    return render_template("tools.html", **_data())


def _render_markdown_page(template_name, key, file_path):
    data = _data()
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as handle:
            data[key] = handle.read()
    else:
        data[key] = ""
    return render_template(template_name, **data)


@app.route("/prehkust.html")
def prehkust():
    return _render_markdown_page("prehkust.html", "PreHKUST", "PreHKUST.md")


@app.route("/hkust.html")
def hkust():
    return _render_markdown_page("hkust.html", "HKUST", "HKUST.md")


def parse_arguments():
    parser = argparse.ArgumentParser(description="AI4QC Portal")
    parser.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Convert the site to static assets",
    )
    parser.add_argument(
        "-b",
        action="store_true",
        default=False,
        dest="build",
        help="Convert the site to static assets",
    )
    parser.add_argument("path", help="Path to the site data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    site_data_path = args.path
    extra_files = main(site_data_path)

    if args.build:
        freezer.freeze()
    else:
        debug_val = os.getenv("FLASK_DEBUG") == "True"
        app.run(port=5000, debug=debug_val, extra_files=extra_files)
