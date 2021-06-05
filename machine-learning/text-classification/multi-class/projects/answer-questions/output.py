import matplotlib.pyplot as plt
import augment

def print_results(raw_data, text_generator, trained):
    plt.plot(trained["history"].history['loss'])
    plt.show()

    print(raw_data["questions"])
    print(augment.augment_each(raw_data["questions"]))
    augmentedData = text_generator.__getitem__(0)
    print(trained["model"].predict(augmentedData))