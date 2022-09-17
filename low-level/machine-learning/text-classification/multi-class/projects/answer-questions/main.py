import augment
import input
import output
import model


raw_data = input.get_training_data()
text_generator = augment.TextAugmentation(raw_data["questions"], raw_data["labels"])
active_model = model.create(raw_data["labels"])
#active_model = input.load_model()
history = model.train(active_model, text_generator)
output.print_results(raw_data, text_generator, active_model, history)
#output.save_model(active_model)
