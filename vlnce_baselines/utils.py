import torch
import torch.distributed as dist
import numpy as np
import math
import copy

import textwrap
import re
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")

from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

class ARGS():
    def __init__(self):
        self.local_rank = 0

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def repeat_allocation(allocations, max_number):
    if torch.is_tensor(max_number):
        max_number = max_number.long().item()
    else:
        max_number = max_number.long()
    allocation_number = len(allocations)
    repeat_time, res = max_number // allocation_number, max_number % allocation_number
    allocations_ = []
    for i in range(repeat_time):
        allocations_ += copy.deepcopy(allocations)
    allocations_ += copy.deepcopy(allocations)[:res]

    return allocations_


def allocate(number, ep_length, size_per_time):
    length_to_indexes = {ep_length[i]: [] for i in
                        range(len(ep_length))}
    for i in range(len(ep_length)):
        length_to_indexes[ep_length[i]] += [i]*number[i]

    values = []
    for i in range(len(number)):
        values += [ep_length[i]] * number[i]

    groups = int((len(values) - 0.01) // size_per_time + 1)
    values.sort(reverse=True)
    load_balance_groups = [[] for grp in range(groups)]

    for v in values:
        load_balance_groups.sort(key=lambda x: sum(x))
        load_balance_groups[0].append(v)

    indexes = []
    set_length = list(set(ep_length))
    for i in range(groups):
        index = np.zeros(len(load_balance_groups[i]),dtype=int)
        for j in range(len(set_length)):
            length_indexes = length_to_indexes[set_length[j]]
            position = np.where(np.array(load_balance_groups[i]) ==
                          set_length[j])[0]
            position_length = len(position)
            index[position] = length_indexes[:position_length]
            length_to_indexes[set_length[j]] = length_indexes[position_length:]
        indexes.append((index).tolist())

    return indexes

def allocate_instructions(instruction_lengths, allocations,ep_length, instruction_ids):
    instruction_ids_copy = copy.deepcopy(instruction_ids)
    allocations_copy = copy.deepcopy(allocations)
    instruction_lengths_copy = copy.deepcopy(instruction_lengths)
    values = []
    value_indexes = []
    weights = []
    for i in range(len(instruction_lengths)):
        instruction_length = instruction_lengths[i]
        values += instruction_length
        value_indexes += len(instruction_length)*[i]
        weights += [ep_length[i]] * len(instruction_length)
    values = np.array(values)
    weights = np.array(weights)
    value_indexes = np.array(value_indexes)
    sorted_index = np.argsort(values*weights)[::-1]
    values = values[sorted_index]
    value_indexes = value_indexes[sorted_index]
    weights = weights[sorted_index]

    groups = len(allocations)
    load_balance_groups = [[] for grp in range(groups)]
    group_weights = [[] for grp in range(groups)]
    instruction_allocations = [[] for grp in range(groups)]
    for j in range(len(values)):
        summation = np.array([np.sum(np.array(load_balance_groups[i])*np.array(group_weights[i])) for i in range(groups)])
        sorted_index = np.argsort(summation)
        for i in sorted_index:
            index = value_indexes[j]
            value = values[j]
            if index in allocations_copy[i]:
                allocations_copy[i].remove(index)
                load_balance_groups[i].append(value)
                group_weights[i].append(weights[j])
                index_in_length = np.where(np.array(instruction_lengths_copy[index]) == value)[0][0]
                instruction_lengths_copy[index].pop(index_in_length)
                instruction_allocations[i].append(instruction_ids_copy[index].pop(index_in_length))
                break

    return instruction_allocations


def allocate_by_scene_for_ddp(number, ep_length, size_per_time):
    length_to_indexes = {ep_length[i]: [] for i in
                        range(len(ep_length))}
    for i in range(len(ep_length)):
        length_to_indexes[ep_length[i]] += [i]*number[i]

    values = []
    for i in range(len(number)):
        values += [ep_length[i]] * number[i]

    groups = int((len(values) - 0.01) // size_per_time + 1)

    values.sort(reverse=True)

    load_balance_groups = [[] for grp in range(groups)]

    for v in values:
        load_balance_groups.sort(key=lambda x: sum(x))
        load_balance_groups[0].append(v)

    indexes = []
    set_length = list(set(ep_length))
    for i in range(groups):
        index = np.zeros(len(load_balance_groups[i]),dtype=int)
        for j in range(len(set_length)):
            length_indexes = length_to_indexes[set_length[j]]
            position = np.where(np.array(load_balance_groups[i]) ==
                          set_length[j])[0]
            position_length = len(position)
            index[position] = length_indexes[:position_length]
            length_to_indexes[set_length[j]] = length_indexes[position_length:]
        indexes.append((index).tolist())

    return indexes

def get_camera_orientations(num_views):
    assert isinstance(num_views, int)
    base_angle_deg = 360 / num_views
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,num_views):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict

def tokenize_(
    sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]
):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def append_text_to_image(image: np.ndarray, text: str, attention: np.ndarray, task: str = "rxr"):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))
    y = 0
    
    # just a sanity check 
    # words = tokenize(text); print(words)
    if attention is None:
        for i, line in enumerate(wrapped_text):
            words = tokenize_(line)
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y += textsize[1] + 10
            
            x = 0
            for word in words:
                wordsize = cv2.getTextSize(
                    word, font, font_size, font_thickness)[0]
                
                cv2.putText(
                    blank_image,
                    word,
                    (x, y),
                    font,
                    font_size,
                    (255, 255, 255),
                    # (0, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA,)
                x += wordsize[0] + 5
    else:
        if task == "rxr":
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            tokenize = tokenizer.tokenize
            # SANITY CHECK
            words = tokenize(text)
            words = ["START"] + words + ["END"]
            word_num = 1
        else:
            words = tokenize_(text)
            word_num = 0
        
        assert len(words) == attention.shape[0], \
            f"sentence len {len(words)} != attention len {attention.shape[0]}"

        
        # print(attention.shape)
        # print(wrapped_text)
        # print(len(wrapped_text), [len(li) for li in wrapped_text])
        attention = attention/np.max(attention)


        for i, line in enumerate(wrapped_text):
            
            if task == "rxr": 
                words = tokenize(line)
            else: 
                words = tokenize_(line)
                
            # print(line, words)
            # print()
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y += textsize[1] + 10
            
            x = 0
            for word in words:
                wordsize = cv2.getTextSize(word, font, font_size, font_thickness)[0]
                weighted_color = int(attention[word_num] * 255)
                # print(i, attention[word_num], weighted_color)
                
                cv2.rectangle(
                    blank_image, 
                    (x, y - wordsize[1]), 
                    (x + wordsize[0], y + wordsize[1]), 
                    (weighted_color, 0, 0),-1)
                
                cv2.putText(
                    blank_image,
                    word,
                    (x, y),
                    font,
                    font_size,
                    (255, 255, 255),
                    # (0, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA,)
                
                x += wordsize[0] + 5
                word_num += 1
            
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)

    return final
