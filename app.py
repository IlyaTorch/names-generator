from flask import Flask, render_template, request
from model.inference import inference
from forms import NameForm

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    names = None
    if request.method == 'POST':
        start_seed = ' ' + request.form['start_seed']
        names = inference(start_seed, 10)

    form = NameForm()

    return render_template('index.html', form=form, names=names)
