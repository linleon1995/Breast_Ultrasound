import importlib


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        # content = fw.readlines()
        content = fw.read().splitlines()
    return content


# def image_loader()