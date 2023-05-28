import logging
import os
from typing import List
import numpy as np
from tqdm import tqdm
from handwtransformer.dataset.Dataset import Dataset
from handwtransformer.dataset.HandwritingSample import HandwritingSample

def parse_csr_file(path: str) -> List[str]:
    """Parses a csr file and returns the text.

    Args:
        path (str): The path to the csr file.

    Returns:
        str: The text in the csr file.
    """
    with open(path) as f:
        lines = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    lines = [line.strip() for line in lines]
    # remove until CSR: + /n line
    lines = lines[lines.index('CSR:') + 2:]
    # remove last line only if it is empty
    if lines[-1] == "":
        lines = lines[:-1]
    return lines

def parse_xml_file(path) -> List[np.ndarray]:
    """Parses an xml file and returns the strokes.

    Args:
        path (str): The path to the xml file.

    Returns:
        List[np.ndarray]: The strokes in the xml file.
    """
    import xml.etree.ElementTree as ET
    # open the xml file with ET 
    with open(path) as f:
        root = ET.parse(f).getroot()

    # iterate over all sub-elements of the root element that are named 'Stroke'
    strokes = []
    for stroke in root.iter('Stroke'):
        # get all the points in the stroke
        points = []
        for point in stroke.iter('Point'):
            points.append((int(point.attrib['x']), int(point.attrib['y'])))
            
        np_points = np.array(points)
        strokes.append(np_points)
    return strokes


def load_iam_from_path(path: str, cache_path: str) -> Dataset:
    """Loads the IAM dataset from a given path. It loads all raw text files in ascii format and their corresponding
    handwriting samples from the xml files.

    Args:
        path (str): The path to the IAM dataset. Should contain ascii, lineStrokes, writers.xml, etc.
        cache_path (str): The path to the cache directory. This directory will be created if it does not exist. It stores a cache of the dataset to speed up loading.

    Returns:
        Dataset: The loaded dataset.
    """
    import xml.etree.ElementTree as ET
    
    dataset = Dataset(path)
    
    dataset.samples = []
    trials = os.listdir(os.path.join(path, 'ascii'))
    # trail is something like 'a01', 'a02', ...
    for trail in tqdm(trials):
        sub_trails = os.listdir(os.path.join(path, 'ascii', trail))
        # sub_trail is something like 'a01-000', 'a01-001', ...
        for sub_trail in sub_trails:
            # not every sub_trail has a corresponding lineStrokes directory
            if not os.path.exists(os.path.join(path, 'lineStrokes', trail, sub_trail)):
                tqdm.write(f'No lineStrokes directory for trail {trail} and sub_trail {sub_trail}.')
                continue
            
            poems = os.listdir(os.path.join(path, 'ascii', trail, sub_trail))
            # remove the .txt ending
            poems = [poem[:-4] for poem in poems]
            handwriting_lines = os.listdir(os.path.join(path, 'lineStrokes', trail, sub_trail))

            # poem is something like 'a01-000u', 'a01-000u', ...
            for poem in poems:
                ascii_lines = parse_csr_file(os.path.join(path, 'ascii', trail, sub_trail, poem + '.txt'))
                
                handwritten_poem_lines = [handwriting_line for handwriting_line in handwriting_lines if handwriting_line.startswith(poem + '-')]
                # not every poem has the right amount of corresponding handwriting lines files
                if len(ascii_lines) != len(handwritten_poem_lines):
                    tqdm.write(f'Poem {poem} has {len(ascii_lines)} ascii lines but {len(handwritten_poem_lines)} handwriting lines.')
                    continue
                
                # handwritten_poem_line is something like 'a01-000u-01.xml', 'a01-000u-02.xml', ...
                for handwritten_poem_line in handwritten_poem_lines:
                    line_index = int(handwritten_poem_line[-6:-4]) - 1

                    # iterate over all sub-elements of the root element that are named 'Stroke'
                    strokes = parse_xml_file(os.path.join(path, 'lineStrokes', trail, sub_trail, handwritten_poem_line))
                                             
                    sample = HandwritingSample(ascii_lines[line_index], strokes)
                    dataset.samples.append(sample)
    
    total_samples = len(dataset.samples)
    total_strokes = sum([len(sample.strokes) for sample in dataset.samples])
    total_data_points = sum([len(stroke) for sample in dataset.samples for stroke in sample.strokes])
    logging.info(f'Loaded {total_samples} samples with {total_strokes} strokes and {total_data_points} data points.')
    
    return dataset