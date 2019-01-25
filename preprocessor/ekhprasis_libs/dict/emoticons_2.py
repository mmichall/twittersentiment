import re

# todo:catch repeating parenthesis
emoticons_original = {
    ':*': '<emo_kiss>',
    ':^*': '<emo_kiss>',
    '😘': '<emo_kiss>',
    ':-*': '<emo_kiss>',
    ':x': '<emo_kiss>',
    ':‑)': '<emo_happy>',
    ':-)': '<emo_happy>',
    ':-))': '<emo_happy>',
    ':-)))': 'very <emo_happy>',
    ':-))))': 'very <emo_happy>',
    ':-)))))': 'very <emo_happy>',
    ':-))))))': 'very <emo_happy>',
    '😃':'<emo_happy>',
    ':)': '<emo_happy>',
    ':))': '<emo_happy>',
    ':)))': 'very <emo_happy>',
    ':))))': 'very <emo_happy>',
    ':)))))': 'very <emo_happy>',
    ':))))))': 'very <emo_happy>',
    ':)))))))': 'very <emo_happy>',
    ':o)': '<emo_happy>',
    ':]': '<emo_happy>',
    ':3': '<emo_happy>',
    ':c)': '<emo_happy>',
    ':>': '<emo_happy>',
    '=]': '<emo_happy>',
    '8)': '<emo_happy>',
    '=)': '<emo_happy>',
    ':}': '<emo_happy>',
    ';‑)': '<emo_wink>',
    ':^)': '<emo_happy>',
    '|;-)': '<emo_wink>',
    ":'-)": '<emo_happy>',
    ":')": '<emo_happy>',
    '\o/': '<emo_happy>',
    '*\\0/*': '<emo_happy>',
    ':-D': '<emo_laugh>',
    ':D': '<emo_laugh>',
    # '(\':': '<emo_laugh>',
    '8-D': '<emo_laugh>',
    '8D': '<emo_laugh>',
    'x-D': '<emo_laugh>',
    'xD': '<emo_laugh>',
    'X-D': '<emo_laugh>',
    'XD': '<emo_laugh>',
    '=-D': '<emo_laugh>',
    '=D': '<emo_laugh>',
    '=-3': '<emo_laugh>',
    '=3': '<emo_laugh>',
    'B^D': '<emo_laugh>',
    ':‑(': '<emo_sad>',
    '😞':'<emo_sad>',
    ":'‑(":'<emo_sad>',
    '>:[': '<emo_sad>',
    ':-(': '<emo_sad>',
    ':-((': '<emo_sad>',
    ':-(((': 'very <emo_sad>',
    ':-((((': 'very <emo_sad>',
    ':-(((((': 'very <emo_sad>',
    ':-((((((': 'very <emo_sad>',
    ':-(((((((': 'very <emo_sad>',
    ':(': '<emo_sad>',
    ':((': '<emo_sad>',
    ':(((': '<emo_sad>',
    ':((((': 'very <emo_sad>',
    ':(((((': 'very <emo_sad>',
    ':((((((': 'very <emo_sad>',
    ':(((((((': 'very <emo_sad>',
    ':((((((((': 'very <emo_sad>',
    '😭': '<emo_sad>',
    ':-c': '<emo_sad>',
    ':c': '<emo_sad>',
    ':-<_': '<emo_sad>',
    ':<_': '<emo_sad>',
    ':-[': '<emo_sad>',
    ':‑[': '<emo_sad>',
    ':[': '<emo_sad>',
    ':{': '<emo_sad>',
    ':-||': '<emo_sad>',
    ':@': '<emo_sad>',
    ":'-(": '<emo_sad>',
    ":'(": '<emo_sad>',
    'D:<_': '<emo_sad>',
    'D:': '<emo_sad>',
    'D8': '<emo_sad>',
    'D;': '<emo_sad>',
    'D=': '<emo_sad>',
    'DX': '<emo_sad>',
    'v.v': '<emo_sad>',
    "D-':": '<emo_sad>',
    '(>_<_)': '<emo_sad>',
    ':|': '<emo_sad>',
    '😲': '<emo_surprise>',
    '🙀': '<emo_surprise>',
    '😧': '<emo_surprise>',
    '>:O': '<emo_surprise>',
    ':-O': '<emo_surprise>',
    ':-o': '<emo_surprise>',
    ':O': '<emo_surprise>',
    'o_O': '<emo_surprise>',
    'o_0': '<emo_surprise>',
    'o.O': '<emo_surprise>',
    'o-o': '<emo_surprise>',
    '8-0': '<emo_surprise>',
    '|-O': '<emo_surprise>',
    '😦': '<emo_surprise>',
    ';))': '<emo_wink>',
    ';-)': '<emo_wink>',
    ';)': '<emo_wink>',
    '*-)': '<emo_wink>',
    '*)': '<emo_wink>',
    ';-]': '<emo_wink>',
    ';]': '<emo_wink>',
    ';D': '<emo_wink>',
    ';^)': '<emo_wink>',
    ':-,': '<emo_wink>',
    '😉': '<emo_wink>',
    '>:P': '<emo_tong>',
    ':-P': '<emo_tong>',
    ':P': '<emo_tong>',
    'X-P': '<emo_tong>',
    ';p': '<emo_tong>',
    'x-p': '<emo_tong>',
    'xp': '<emo_tong>',
    'XP': '<emo_tong>',
    ':-p': '<emo_tong>',
    ':p': '<emo_tong>',
    '😜': '<emo_tong>',
    '🤗': '<emo_happy>',
    '=p': '<emo_tong>',
    ':-?': '<emo_tong>',
    ':?': '<emo_tong>',
    ':-b': '<emo_tong>',
    '😛': '<emo_tong>',
    ':b': '<emo_tong>',
    ':-&': '<emo_tong>',
    '😋': '<emo_tong>',
    '😐': '<emo_annoyed>',
    '>:\\': '<emo_annoyed>',
    '😏': '<emo_annoyed>',
    '>:/': '<emo_annoyed>',
    ':\\': '<emo_annoyed>',
    '. __ .': '<emo_annoyed>',
    '=/': '<emo_annoyed>',
    ':‑/': '<emo_annoyed>',
    '-_-': '<emo_annoyed>',
    '=\\': '<emo_annoyed>',
    ':-/': '<emo_annoyed>',
    ':-.': '<emo_annoyed>',
    ':/': '<emo_annoyed>',
    ':\\': '<emo_annoyed>',
    '=/': '<emo_annoyed>',
    '=\\': '<emo_annoyed>',
    '😑': '<emo_annoyed>',
    ':L': '<emo_annoyed>',
    '=L': '<emo_annoyed>',
    ':S': '<emo_annoyed>',
    '>.<_': '<emo_annoyed>',
    ':-|': '<emo_annoyed>',
    '<_:-|': '<emo_annoyed>',
    '😒': '<emo_annoyed>',
    '😖': '<emo_seallips>',
    ':-X': '<emo_seallips>',
    ':X': '<emo_seallips>',
    ':-#': '<emo_seallips>',
    ':#': '<emo_seallips>',
    '😇': '<emo_angel>',
    'O:-)': '<emo_angel>',
    '0:-3': '<emo_angel>',
    '0:3': '<emo_angel>',
    '0:-)': '<emo_angel>',
    '0:)': '<emo_angel>',
    '0;^)': '<emo_angel>',
    '>:)': '<emo_devil>',
    '>:D': '<emo_devil>',
    '>:-D': '<emo_devil>',
    '>;)': '<emo_devil>',
    '>:-)': '<emo_devil>',
    '}:-)': '<emo_devil>',
    '}:)': '<emo_devil>',
    '3:-)': '<emo_devil>',
    '3:)': '<emo_devil>',
    'o/\o': '<emo_highfive>',
    '^5': '<emo_highfive>',
    '>_>^': '<emo_highfive>',
    '^<__<_': '<emo_highfive>',  # todo:fix tokenizer - MISSES THIS
    '<_3': '<emo_heart>',
    '>😃': '<emo_happy>',
    '😂':'<emo_laugh>',
    '>😁': '<emo_laugh>',
    '>:-D': '<emo_happy>',
    '>😉': '<emo_wink>',
    ';‑]': '<emo_wink>',
    '>:-)': '<emo_happy>',
    '}:-)': '<emo_happy>',
    '}😃': '<emo_happy>',
    '3:-)': '<emo_happy>',
    '3😃': '<emo_happy>',
    'o/\o': '<emo_highfive>',
    '^5': '<emo_highfive>',
    '>_>^': '<emo_highfive>',
    '^<__<_': '<emo_highfive>',  # todo:fix tokenizer - MISSES THIS
    '<_3': '<emo_heart>',
    '😿': '<emo_sad>',
    '😨':'<emo_sad>',
    '😹': '<emo_laugh>',
    '😽': '<emo_kiss>',
    '😡': '<emo_angry>',
    '😚': '<emo_kiss>',
    '😰': '<emo_sad>',
    '😻': '<emo_love>',
    '💖': '<emo_heart>',
    '0;^)': '<emo_wink>',
    '😩': '<emo_sad>',
    '😍': '<emo_love>',
    '😢': '<emo_sad>',
    '😎': '<emo_happy>',
    '😕': '<emo_sad>',
    '😀': '<emo_laugh>',
    '😅': '<emo_laugh>',
    '🙂': '<emo_happy>',
    '🙁': '<emo_sad>',
    '😺':'<emo_happy>',
    '😄':'<emo_laugh>',
    '😁':'<emo_laugh>',
    '😣': '<emo_sad>',
    '😆':'<emo_laugh>',
    '😴':'<emo_sleep>',
    '❤':'<emo_heart>',
    '😸':'<emo_happy>',
    '😾':'<emo_sad>',
    '💔':'<emo_brokeheart>',
    '😪': '<emo_sleep>',
    '😊': '<emo_happy>',
    '^_^': '<emo_happy>',
    '💙': '<emo_heart>',
    '💕': '<emo_heart>',
    '😫': '<emo_sad>',
    '😝': '<emo_tong>',
    '😤': '<emo_angry>',
    '😠': '<emo_angry>',
    '🤪': '<emo_happy>',
    ';-;': '<emo_sad>',
    '>:(': '<emo_sad>',
    '😟': '<emo_sad>',
    '😔': '<emo_sad>',
    ':‑d': '<emo_happy>',
    '(^_^)': '<emo_happy>',
    ':‑c': '<emo_sad>',
    ":'‑)":'<emo_happy>',
    '😥': '<emo_sad>',
    '😓': '<emo_sad>',
    '☺': '<emo_happy>',
    '❤': '<emo_heart>',
    '💛': '<emo_heart>',
    '(^.^)': '<emo_happy>',
    ':,)': '<emo_happy>',
    '💘': '<emo_heart>',
    '☺': '<emo_happy>'

}

# todo: clear this mess
pattern = re.compile("^[:=\*\-\(\)\[\]x0oO\#\<emo_\>8\\.\'|\{\}\@]+$")
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

        if "<emo_" in mirror:
            mirror = mirror.replace("<emo_", ">")
        elif ">" in mirror:
            mirror = mirror.replace(">", "<emo_")

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
    "positive": {'<emo_highfive>', '<emo_laugh>', '<emo_heart>', '<emo_happy>'},
    "negative": {'<emo_annoyed>', '<emo_sad>', }
}


def print_positive(sentiment):
    for e, tag in emoticons_original.items():
        if tag in emoticon_groups[sentiment]:
            print(e)

# print_positive("negative")
# print(" ".join(list(emoticons.keys())))
# [print(e) for e in list(emoticons.keys())]