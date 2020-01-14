from protein_transformer.losses import inverse_trig_transform
from protein_transformer.losses import angles_to_coords

class ModelPrediction(object):
    """ Represents a prediction from a model, can be transformed as needed. """
    def __init__(self, input_sequence, raw_model_output, modality="sincos"):
        self.input_sequence = input_sequence
        self.raw_model_output = raw_model_output
        self.modality = "sincos"
        self.data = raw_model_output

    def to_radians(self):
        """ Modifies data to angles. Returns data. """
        if self.modality == "radians":
            return self.data
        elif self.modality == "sincos":
            self.data = inverse_trig_transform(self.data)
            self.modality = "radians"
            return self.data
        else:
            raise NotImplementedError

    def to_coordinates(self):
        """ Modifies data to coordinates. Returns data. """
        if self.modality == "coords":
            return self.data
        else:
            self.to_radians()
            self.data = angles_to_coords(self.data)
            self.modality = "coords"
            return self.data


