import uproot
import numpy as np

class Sample(object):
    def __init__(self, name, path, x_sec, total_weight, sample_weight):
        self.name = name
        self.path = path
        self.x_sec = x_sec
        self.total_weight = total_weight
        self.sample_weight = sample_weight
        self.weights = np.array([])
        self.mask = np.array([])
        self.m_fit = np.array([])

    def show(self):
        print("{0} (x_sec = {1:2.2f}, weight = {2:2.2f})"
              .format(self.name, self.x_sec, self.sample_weight))

    def get_events(self):
        try: self.events = uproot.open(self.path)["Events"]
        except AttributeError:
            print("ERROR: failed to open file {0:s}".format(self.path))
        self.n_entries = self.events.numentries
        self.weights = np.ones(self.n_entries)
        self.mask = np.ones(self.n_entries, dtype=bool)
        self.m_fit = np.zeros(self.n_entries)

    def parse_categories(self, categories, evt_cat_array):
        self.cats = np.array([categories[cat] for cat in evt_cat_array])
        self.ll   = np.array([cat[:2] for cat in self.cats])
        self.tt   = np.array([cat[2:] for cat in self.cats])
