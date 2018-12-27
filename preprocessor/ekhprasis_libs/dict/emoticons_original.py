import re

# todo:catch repeating parenthesis
emoticons_original = {
    ':*': '<kiss>',
    ':^*': '<kiss>',
    '😘': 'kiss',
    ':-*': '<kiss>',
    ':x': '<kiss>',
    ':‑)': '<happy>',
    ':-)': '<happy>',
    ':-))': '<happy>',
    ':-)))': '<happy><happy>',
    ':-))))': '<happy><happy>',
    ':-)))))': '<happy><happy>',
    ':-))))))': '<happy><happy>',
    '😃':'<happy>',
    ':)': '<happy>',
    ':))': '<happy>',
    ':)))': '<happy><happy>',
    ':))))': '<happy><happy>',
    ':)))))': '<happy><happy>',
    ':))))))': '<happy><happy>',
    ':)))))))': '<happy><happy>',
    ':o)': '<happy>',
    ':]': '<happy>',
    ':3': '<happy>',
    ':c)': '<happy>',
    ':>': '<happy>',
    '=]': '<happy>',
    '8)': '<happy>',
    '=)': '<happy>',
    ':}': '<happy>',
    ';‑)': '<wink>',
    ':^)': '<happy>',
    '|;-)': '<wink>',
    ":'-)": '<happy>',
    ":')": '<happy>',
    '\o/': '<happy>',
    '*\\0/*': '<happy>',
    ':-D': '<laugh>',
    ':D': '<laugh>',
    # '(\':': '<laugh>',
    '8-D': '<laugh>',
    '8D': '<laugh>',
    'x-D': '<laugh>',
    'xD': '<laugh>',
    'X-D': '<laugh>',
    'XD': '<laugh>',
    '=-D': '<laugh>',
    '=D': '<laugh>',
    '=-3': '<laugh>',
    '=3': '<laugh>',
    'B^D': '<laugh>',
    ':‑(': '<sad>',
    '😞':'<sad>',
    ":'‑(":'<sad>',
    '>:[': '<sad>',
    ':-(': '<sad>',
    ':-((': '<sad>',
    ':-(((': '<sad><sad>',
    ':-((((': '<sad><sad>',
    ':-(((((': '<sad><sad>',
    ':-((((((': '<sad><sad>',
    ':-(((((((': '<sad><sad>',
    ':(': '<sad>',
    ':((': '<sad>',
    ':(((': '<sad>',
    ':((((': '<sad><sad>',
    ':(((((': '<sad><sad>',
    ':((((((': '<sad><sad>',
    ':(((((((': '<sad><sad>',
    ':((((((((': '<sad><sad>',
    '😭': '<sad>',
    ':-c': '<sad>',
    ':c': '<sad>',
    ':-<': '<sad>',
    ':<': '<sad>',
    ':-[': '<sad>',
    ':‑[': '<sad>',
    ':[': '<sad>',
    ':{': '<sad>',
    ':-||': '<sad>',
    ':@': '<sad>',
    ":'-(": '<sad>',
    ":'(": '<sad>',
    'D:<': '<sad>',
    'D:': '<sad>',
    'D8': '<sad>',
    'D;': '<sad>',
    'D=': '<sad>',
    'DX': '<sad>',
    'v.v': '<sad>',
    "D-':": '<sad>',
    '(>_<)': '<sad>',
    ':|': '<sad>',
    '😲': '<surprise>',
    '🙀': '<surprise>',
    '😧': '<surprise>',
    '>:O': '<surprise>',
    ':-O': '<surprise>',
    ':-o': '<surprise>',
    ':O': '<surprise>',
    'o_O': '<surprise>',
    'o_0': '<surprise>',
    'o.O': '<surprise>',
    'o-o': '<surprise>',
    '8-0': '<surprise>',
    '|-O': '<surprise>',
    '😦': '<surprise>',
    ';))': '<wink>',
    ';-)': '<wink>',
    ';)': '<wink>',
    '*-)': '<wink>',
    '*)': '<wink>',
    ';-]': '<wink>',
    ';]': '<wink>',
    ';D': '<wink>',
    ';^)': '<wink>',
    ':-,': '<wink>',
    '😉': '<wink>',
    '>:P': '<tong>',
    ':-P': '<tong>',
    ':P': '<tong>',
    'X-P': '<tong>',
    ';p': '<tong>',
    'x-p': '<tong>',
    'xp': '<tong>',
    'XP': '<tong>',
    ':-p': '<tong>',
    ':p': '<tong>',
    '😜': '<tong>',
    '🤗': '<happy>',
    '=p': '<tong>',
    ':-?': '<tong>',
    ':?': '<tong>',
    ':-b': '<tong>',
    '😛': '<tong>',
    ':b': '<tong>',
    ':-&': '<tong>',
    '😋': '<tong>',
    '😐': '<annoyed>',
    '>:\\': '<annoyed>',
    '😏': '<annoyed>',
    '>:/': '<annoyed>',
    ':\\': '<annoyed>',
    '. __ .': '<annoyed>',
    '=/': '<annoyed>',
    ':‑/': '<annoyed>',
    '-_-': '<annoyed>',
    '=\\': '<annoyed>',
    ':-/': '<annoyed>',
    ':-.': '<annoyed>',
    ':/': '<annoyed>',
    ':\\': '<annoyed>',
    '=/': '<annoyed>',
    '=\\': '<annoyed>',
    '😑': '<annoyed>',
    ':L': '<annoyed>',
    '=L': '<annoyed>',
    ':S': '<annoyed>',
    '>.<': '<annoyed>',
    ':-|': '<annoyed>',
    '<:-|': '<annoyed>',
    '😒': '<annoyed>',
    '😖': '<seallips>',
    ':-X': '<seallips>',
    ':X': '<seallips>',
    ':-#': '<seallips>',
    ':#': '<seallips>',
    '😇': '<angel>',
    'O:-)': '<angel>',
    '0:-3': '<angel>',
    '0:3': '<angel>',
    '0:-)': '<angel>',
    '0:)': '<angel>',
    '0;^)': '<angel>',
    '>:)': '<devil>',
    '>:D': '<devil>',
    '>:-D': '<devil>',
    '>;)': '<devil>',
    '>:-)': '<devil>',
    '}:-)': '<devil>',
    '}:)': '<devil>',
    '3:-)': '<devil>',
    '3:)': '<devil>',
    'o/\o': '<highfive>',
    '^5': '<highfive>',
    '>_>^': '<highfive>',
    '^<_<': '<highfive>',  # todo:fix tokenizer - MISSES THIS
    '<3': '<heart>',
    '>😃': '<happy>',
    '😂':'<laugh>',
    '>😁': '<laugh>',
    '>:-D': '<happy>',
    '>😉': '<wink>',
    ';‑]': '<wink>',
    '>:-)': '<happy>',
    '}:-)': '<happy>',
    '}😃': '<happy>',
    '3:-)': '<happy>',
    '3😃': '<happy>',
    'o/\o': '<highfive>',
    '^5': '<highfive>',
    '>_>^': '<highfive>',
    '^<_<': '<highfive>',  # todo:fix tokenizer - MISSES THIS
    '<3': '<heart>',
    '😿': '<sad>',
    '😨':'<sad>',
    '😹': '<laugh>',
    '😽': '<kiss>',
    '😡': '<angry>',
    '😚': '<kiss>',
    '😰': '<sad>',
    '😻': '<love>',
    '💖': '<heart>',
    '0;^)': '<wink>',
    '😩': '<sad>',
    '😍': '<love>',
    '😢': '<sad>',
    '😎': '<happy>',
    '😕': '<sad>',
    '😀': '<laugh>',
    '😅': '<laugh>',
    '🙂': '<happy>',
    '🙁': '<sad>',
    '😺':'<happy>',
    '😄':'<laugh>',
    '😁':'<laugh>',
    '😣': '<sad>',
    '😆':'<laugh>',
    '😴':'<sleep>',
    '❤':'<heart>',
    '😸':'<happy>',
    '😾':'<sad>',
    '💔':'<brokeheart>',
    '😪': '<sleep>',
    '😊': '<happy>',
    '^_^': '<happy>',
    '💙': '<heart>',
    '💕': '<heart>',
    '😫': '<sad>',
    '😝': '<tong>',
    '😤': '<angry>',
    '😠': '<angry>',
    '🤪': '<happy>',
    ';-;': '<sad>',
    '>:(': '<sad>',
    '😟': '<sad>',
    '😔': '<sad>',
    ':‑d': 'happy',
    '(^_^)': '<happy>',
    ':‑c': '<sad>',
    ":'‑)":'<happy>',
    '😥': '<sad>',
    '😓': '<sad>'

}

# todo: clear this mess
pattern = re.compile("^[:=\*\-\(\)\[\]x0oO\#\<\>8\\.\'|\{\}\@]+$")
mirror_emoticons = {}
for exp, tag in emoticons_original.items():
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
emoticons_original.update(mirror_emoticons)

for exp, tag in list(emoticons_original.items()):
    if exp.lower() not in emoticons_original:
        emoticons_original[exp.lower()] = tag

emoticon_groups = {
    "positive": {'<highfive>', '<laugh>', '<heart>', '<happy>'},
    "negative": {'<annoyed>', '<sad>', }
}


def print_positive(sentiment):
    for e, tag in emoticons_original.items():
        if tag in emoticon_groups[sentiment]:
            print(e)

# print_positive("negative")
# print(" ".join(list(emoticons.keys())))
# [print(e) for e in list(emoticons.keys())]