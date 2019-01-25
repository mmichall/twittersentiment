import re

# todo:catch repeating parenthesis
emoticons = {
    ':*': '😘',
    ':^*': '😘',
    ':-*': '😘',
    '(^○^)': '😃',
    ':-)': '😃',
    ':-))': '😃',
    ':-)))': '😃😃',
    ':-))))': '😃😃',
    ':-)))))': '😃😃',
    ':-))))))': '😃😃',
    '>:)': '😞',
    '>:(': '😃',
    '^_^': '😃',
    '^^': '😃',
    '(^�^)': '😃',
    ':)': '😃',
    'ツ': '😃',
    ':))': '😃',
    ':)))': '😃',
    ':))))': '😃😃',
    ':)))))': '😃😃',
    ':))))))': '😃😃',
    ':)))))))': '😃😃',
    '😲)': '😃',
    ':]': '😃',
    ':3': '😃',
    ':c)': '😃',
    ':>': '😃',
    '=]': '😃',
    '8)': '😃',
    '=)': '😃',
    ':}': '😃',
    ':^)': '😃',
    '|;-)': '😃',
    ":'-)": '😃',
    ":')": '😃',
    '\o/': '😃',
    '*\\0/*': '😃',
    ':-D': '😁',
    ':D': '😁',
    # '(\':': '😁',
    '8-D': '😁',
    '8D': '😁',
    'x-D': '😁',
    'xD': '😁',
    'X-D': '😁',
    'XD': '😁',
    '=-D': '😁',
    '=D': '😁',
    '=-3': '😁',
    '=3': '😁',
    'B^D': '😁',
    '>:[': '😞',
    ':-(': '😞',
    '-.-': '😞',
    ':-((': '😞',
    ':-(((': '😞😞',
    ':-((((': '😞😞',
    ':-(((((': '😞😞',
    ':-((((((': '😞😞',
    ':-(((((((': '😞😞',
    ':(': '😞',
    ';-;': '😞',
    ":'‑(": '😞',
    ':((': '😞',
    ':(((': '😞',
    ':((((': '😞😞',
    ':(((((': '😞😞',
    ':((((((': '😞😞',
    ':(((((((': '😞😞',
    ':((((((((': '😞😞',
    ':-c': '😞',
    ':c': '😞',
    ':-<': '😞',
    ':<': '😞',
    ':-[': '😞',
    ':[': '😞',
    ':{': '😞',
    ':-||': '😞',
    ':@': '😞',
    ":'-(": '😞',
    ":'(": '😞',
    'D:<': '😞',
    'D:': '😞',
    'D8': '😞',
    'D;': '😞',
    'D=': '😞',
    'DX': '😞',
    'v.v': '😞',
    "D-':": '😞',
    '(>_<)': '😞',
    ':|': '😞',
    '>:O': '😲',
    ':-O': '😲',
    ":'o": '😲',
    ':-o': '😲',
    ':O': '😲',
    '°o°': '😲',
    'o_O': '😲',
    'o_0': '😲',
    'o.O': '😲',
    'o-o': '😲',
    '8-0': '😲',
    '|-O': '😲',
    ';-)': '😉',
    ';)': '😉',
    '*-)': '😉',
    '*)': '😉',
    ';-]': '😉',
    ';]': '😉',
    ';")': '😉',
    ';D': '😉',
    ';^)': '😉',
    ':-,': '😉',
    '>:P': '😋',
    ':-P': '😋',
    ':P': '😋',
    'X-P': '😋',
    'x-p': '😋',
    'xp': '😋',
    'XP': '😋',
    ':-p': '😋',
    ':p': '😋',
    '=p': '😋',
    ':-Þ': '😋',
    ':Þ': '😋',
    ':-b': '😋',
    ':b': '😋',
    ':-&': '😋',
    '>:\\': '😏',
    '>:/': '😏',
    '. __ .': '😏',
    '://': '😏',
    ':-/': '😏',
    ':-\\': '😏',
    ':-.': '😏',
    ':/': '😏',
    ':‑/': '😏',
    ':\\': '😏',
    '=/': '😏',
    '-_-': '😏',
    '=\\': '😏',
    ':L': '😏',
    '=L': '😏',
    ':S': '😏',
    '>.<': '😏',
    ':-|': '😏',
    '<:-|': '😏',
    ':-X': '😖',
    ':X': '😖',
    ':-#': '😖',
    ':#': '😖',
    'O:-)': 'O:-)',
    '0:-3': 'O:-)',
    '0:3': 'O:-)',
    '0:-)': 'O:-)',
    '0😃': 'O:-)',
    '0;^)': 'O:-)',
    '>😃': '>:-)',
    '>😁': '>:-)',
    '>:-D': '>:-)',
    '>😉': '>:-)',
    '>:-)': '>:-)',
    '}:-)': '>:-)',
    '}😃': '>:-)',
    '3:-)': '>:-)',
    '3😃': '>:-)',
    'o/\o': '<_highfive>',
    '^5': 'emotionhighfive>',
    '>_>^': 'emotionhighfive>',
    '^<_<': 'emotionhighfive>',  # todo:fix tokenizer - MISSES THIS
    '<3': '❤',
    '😿': '😢',
    '😹': '😂',
    '😽': '😘',
    '😚': '😘',
    '💖': '❤',
    '(^.^)': '😃',
    ':///': '😏'
}

# todo: clear this mess
pattern = re.compile("^[:=\*\-\(\)\[\]x0oO\#\<\>8\\.\'|\{\}\@]+$")
mirror_emoticons = {}
for exp, tag in emoticons.items():
    if pattern.match(exp) \
            and any(ext in exp for ext in [";", ":", "="]) \
            and not any(ext in exp for ext in ["L", "D", "p", "P", "3"]):
        mirror = exp[::-1]

        if "{" in mirror:
            mirror = mirror.replace("{", "}")
        elif "}" in mirror:
            mirror = mirror.replace("}", "{")

        if "(" in mirror:
            mirror = mirror.replace("(", ")")
        elif ")" in mirror:
            mirror = mirror.replace(")", "(")

        if "<" in mirror:
            mirror = mirror.replace("<", ">")
        elif ">" in mirror:
            mirror = mirror.replace(">", "<")

        if "[" in mirror:
            mirror = mirror.replace("[", "]")
        elif "]" in mirror:
            mirror = mirror.replace("]", "[")

        if "\\" in mirror:
            mirror = mirror.replace("\\", "/")
        elif "/" in mirror:
            mirror = mirror.replace("/", "\\")

        # print(exp + "\t\t" + mirror)
        mirror_emoticons[mirror] = tag
emoticons.update(mirror_emoticons)

for exp, tag in list(emoticons.items()):
    if exp.lower() not in emoticons:
        emoticons[exp.lower()] = tag

emoticon_groups = {
    "positive": {'<highfive>', '😁', '<heart>', '😃'},
    "negative": {'😏', '😞', }
}


def print_positive(sentiment):
    for e, tag in emoticons.items():
        if tag in emoticon_groups[sentiment]:
            print(e)

# print_positive("negative")
# print(" ".join(list(emoticons.keys())))
# [print(e) for e in list(emoticons.keys())]
