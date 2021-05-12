import re
import os


def process(file, process_file):
    for sentence in file:
        # x = re.sub(r"\⌊.*?\⌋", "", sentence)  # link - keep
        s = re.sub(r"\[.*?\] |\|", "", sentence)  # remove id number
        s_title = re.sub(r"\⌊\δ.*?\δ\⌋", "", s)  # title - remove
        s_link = re.sub(r"\⌊\>|\>\⌋", "", s_title)  # link - keep
        s_heading = re.sub(r"\⌊\=.*?\=\⌋", "", s_link)  # heading - remove
        s_bold = re.sub(r"\⌊\∗|\∗\⌋", "", s_heading)  # bold - keep
        s_italic = re.sub(r"\⌊\/|\/\⌋", "", s_bold)  # italic - keep
        s_underline = re.sub(r"\⌊\_|\_\⌋", "", s_italic)  # underline - keep
        s_quote = re.sub(r"\⌊\"|\"\⌋", "", s_italic)  # quote - keep
        s_list = re.sub(r"\⌊\•|\•\⌋", "", s_quote)  # list - keep
        s_list_item = re.sub(r"\⌊\#|\#\⌋", "", s_list)  # list item - keep
        s_big = re.sub(r"\⌊\↑|\↑\⌋", "", s_list_item)  # big text - keep
        s_small = re.sub(r"\⌊\↓|\↓\⌋", "", s_big)  # small text - keep
        s_final = re.sub(r"\⌊.*?\⌋", "", s_small)  # Remove everything else.

        process_file.write(s_final)


gml_dir_path = '../big_training/gml'
processed_dir_path = '../big_training/processed'

gml_paths = os.listdir(gml_dir_path)

for gml_path in gml_paths:
    print(gml_path)
    gml_file = open(gml_dir_path + '/' + gml_path, 'r', encoding='utf-8')
    # Create file in processed directory
    processed_file_name = gml_path[:-3]
    processed_file = open(os.path.join(processed_dir_path, processed_file_name), 'w', encoding='utf-8')
    process(gml_file, processed_file)