import mido
import numpy as np
from PIL import Image
import os


class MidiFile(mido.MidiFile):

    def __init__(self, filename):
        mido.MidiFile.__init__(self, filename)
        print(self)
        self.msgs_matrix = []
        self.notes = np.array
        self.notes_full = np.array

    def get_msgs(self, verbose=0):
        msgs = []
        index = 0
        time_counter = 0
        note_type = 1
        msgs_arr = []
        for msg in self:
            msgs.append(msg)
            if verbose == 1:
                print(msg)
        for msg in msgs:
            if msg.type == "note_on":
                if msg.velocity == 0:
                    note_type = 0
                    note = msg.note
                else:
                    note_type = 1
                    note = msg.note
            elif msg.type == "note_off":
                note_type = 0
                note = msg.note
            else:
                time_counter += msg.time
                continue
            time_counter += msg.time
            index += 1
            msgs_arr.append([index, note_type, note, time_counter])
            self.msgs_matrix = np.array(msgs_arr).astype("float32")
        if verbose == 1:
            for msg in msgs:
                print(msg)
    def get_notes(self, note_rate=8, lenght=100, width=128, x=-10):
        stop = 0
        notes = []
        for msg in self.msgs_matrix:
            stop += msg[3]
            if msg[1] == 1:
                note = msg[2]
                start = msg[3]
                for msg_off in self.msgs_matrix[int(msg[0]):, :]:
                    if msg_off[1] == 0 and msg_off[2] == note:
                        stop = msg_off[3]
                        break
                notes.append([note, start, stop])

        self.notes = np.array(notes)
        self.notes[:, 2] = self.notes[:, 2] * note_rate
        self.notes[:, 1] = self.notes[:, 1] * note_rate
        new_width_rate = (int(self.notes[-1, 2]) // width) + 1
        self.notes_full = np.zeros((lenght, new_width_rate * width))
        for note in self.notes:
            self.notes_full[int(note[0] + x), int(note[1]):int(note[2]) + 1] = 255

    def export_png(self, path, width=128):
        x = 0
        while x < self.notes_full.shape[1]:
            im = Image.fromarray(self.notes_full[:, x:x + width])
            im = im.convert("L")
            im.save(path + f"\\{int(x / width)}.png")
            x += width

    def complete_task(self, path, verbose=0, note_rate=8, length=100, x=-10, width=128):
        self.get_msgs(verbose)
        self.get_notes(note_rate, length, width, x)
        self.export_png(path, width)


tracks = os.listdir(os.getcwd() + "\\tracks")
for track in tracks:
    dirName = os.getcwd() + "\\exported\\" + f"{track}"
    try:
        os.mkdir(dirName)
    except FileExistsError:
        print("Directory ", dirName, " already exists")
        continue
    try:
        file = MidiFile(os.getcwd() + "\\tracks\\" + track)
        file.complete_task(path=os.getcwd() + f"\\exported\\{track}")
    except:
        print(f"exception in {track}")
