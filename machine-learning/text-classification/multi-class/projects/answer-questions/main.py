import augment
import input
import output
import model

raw_data = input.get_training_data()
text_generator = augment.TextAugmentation(raw_data["questions"], raw_data["labels"])
trained = model.train(text_generator)
output.print_results(raw_data, text_generator, trained)
