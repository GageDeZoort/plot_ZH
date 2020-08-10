import uproot
import pickle
import numpy as np

class Sample(object):
    def __init__(self, name, path, x_sec, total_weight, sample_weight, 
                 lookup_path="../lookup_tables"):
        self.name = name
        self.path = path
        self.x_sec = x_sec
        self.total_weight = total_weight
        self.sample_weight = sample_weight
        self.weights = np.array([])
        self.mask = np.array([])
        self.mtt_fit = np.array([])
        self.m4l = np.array([])
        self.mA = np.array([])
        self.mA_c = np.array([])
        self.get_events()
        self.lookup_path = lookup_path
        self.lookup_table = {}
        try: 
            lookup_table_file = open("{0}/{1}_masses.pkl"
                                     .format(lookup_path, self.name), 'rb')
            self.lookup_table = pickle.load(lookup_table_file)
            lookup_table_file.close()
        except: 
            print("WARNING: creating new lookup table for {0}"
                  .format(self.name))
        self.n_recalculated = 0


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
        self.mtt_fit = np.zeros(self.n_entries)
        self.m4l     = np.zeros(self.n_entries)
        self.mA      = np.zeros(self.n_entries)
        self.mA_c    = np.zeros(self.n_entries)
        
    def parse_categories(self, categories, evt_cat_array):
        self.cats = np.array([categories[cat] for cat in evt_cat_array])
        self.ll   = np.array([cat[:2] for cat in self.cats])
        self.tt   = np.array([cat[2:] for cat in self.cats])

    def write_lookup_table(self):
        with open("{0}/{1}_masses.pkl"
                  .format(self.lookup_path, self.name), 'wb') as f:
            pickle.dump(self.lookup_table, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
