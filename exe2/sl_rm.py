from joblib import dump, load
import json
import os
from configs import *

# for model/result saving/loading

def save_model(model, model_name, paras):
    # model 为实际的模型，model_name为采用的方法名，比如PCA，LDA，paras为字典表示的方法采用的参数
    model_name = build_file_name(model_name, paras, MODEL)
    dump(model, model_name)


def load_model(model_name, paras):
    model_name = build_file_name(model_name, paras, MODEL)
    model = None
    if not os.path.exists(model_name):
        return model
    model = load(model_name)
    return model


def save_result(content, model_name, paras={}):
    # content是一个模型评估记录, 后两个参数同上
    result_name = build_file_name(model_name, paras, RESULT)
    with open(result_name, 'w') as f:
        f.write(json.dumps(content))


def load_result(model_name="", paras={}, file_name=""):
    if file_name == "":
        result_name = build_file_name(model_name, paras, RESULT)
    else:
        result_name = model_name+"/"+file_name
    content = None
    if not os.path.exists(result_name):
        return content
    with open(result_name, 'r') as f:
        content = json.loads(f.read())
    return content


def build_file_name(model_name, paras, typ):
    if not os.path.exists(typ):
        os.mkdir(typ)
    os.chdir(typ)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir("..")

    file_name = "_".join([str(p) for p in paras.items()])

    if typ == MODEL:
        file_name = "{}/{}/{}.joblib".format(typ, model_name, file_name)
    elif typ == RESULT:
        file_name = "{}/{}/{}.txt".format(typ, model_name, file_name)
    return file_name


if __name__ == "__main__":
    save_model({"test": 1}, "test", {"C": 1})
    print(load_model("test", {"C": 1}))
    save_result({"sc": 1}, "test", {"C": 1})
    print(load_result("test", {"C": 1}))
    print(load_model("not_exist", {}))
    print(load_result("not_exist", {}))
