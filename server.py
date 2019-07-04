from flask import Flask, request, render_template, url_for
from predictors import head_line

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form
    status = head_line(form_data["headline"])
    return render_template("response.html",status=status)

if __name__ == "__main__":
    app.run()
