from wtforms import Form, StringField


class NameForm(Form):
    start_seed = StringField('Start Seed')
