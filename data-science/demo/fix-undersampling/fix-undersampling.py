import pandas as pd
import numpy as np
import csv

def read_csv():
    csv = pd.read_csv('data/original.csv', delimiter=';')
    headers = list(csv.columns.values)
    return {
        "items": csv.to_dict('records'),
        "headers": headers
    }

def write_csv(items, headers):
    with open("data/processed.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        for item in items:
            writer.writerow(item.values())

def fix(items):
    labels_count = get_labels_count(items)
    max_count = get_max_count(labels_count)
    for label in labels_count:
        new_samples_count = max_count - labels_count[label]
        samples = create_samples(items, label, new_samples_count)
        if samples:
            items.extend(samples)
    items = sorted(items, key=lambda k: k['Label'])
    return items

def create_samples(items, expected_label, new_samples_count):
    if new_samples_count == 0:
        return
    actual_items = items_with_label(items, expected_label)
    new_items = []
    added_samples = 0
    while added_samples < new_samples_count:
        for item in actual_items:
            if (added_samples >= new_samples_count):
                break
            new_items.append(item)
            added_samples += 1
    return new_items

def items_with_label(items, expected_label):
    with_label = []
    for item in items:
        label = item["Label"]
        if label == expected_label:
            with_label.append(item)
    return with_label

def get_labels_count(items):
    count = {}
    for item in items:
        label = item["Label"]
        if label in count:
            count[label] = count[label] + 1
        else:
            count[label] = 1
    return count

def get_max_count(labels_count):
    max_key = max(labels_count, key=labels_count.get)
    return labels_count[max_key]

csv_data = read_csv()
items = fix(csv_data["items"])
write_csv(items, csv_data["headers"])
