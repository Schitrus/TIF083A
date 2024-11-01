def read_data(filepath):
    """
    Reads and parses the data of tsv file
    Returns the object names, time points, list of objects
        Objects has the structure of (x-coordinates, y-coordinates, z-coordinates)

    Arguments:
    filepath -- filepath to the tsv file
    """
    f = open(filepath)

    # Read lines until header appears
    obj_names = []
    header = []
    raw_data = []
    while True:
        line = f.readline()
        if line.startswith('DATA_TYPES') or line.startswith('MARKER_NAMES'):
            obj_names = line.split('\t')[1:]
        elif line.startswith('Frame'):
            header = line.split('\t')
            break
        elif line[0].isdigit(): # incase there is no header at all
            header = line.split('\t')
            raw_data.append(line)
            break
        
    num_of_objects = (len(header) - 2)//3

    # Parse the rest, seperated by lines
    for x in f:
        raw_data.append(x)

    # Init data
    ts = []
    objs = []
    for i in range(num_of_objects):
        objs.append(([], [], []))

    # Parse each raw line as time and for each object x, y, z
    for data in raw_data:
        split_data = data.split('\t')
        ts.append(float(split_data[1]))
        for i in range(num_of_objects):
            objs[i][0].append(float(split_data[2 + 3*i]))
            objs[i][1].append(float(split_data[3 + 3*i]))
            objs[i][2].append(float(split_data[4 + 3*i]))

    return obj_names, ts, objs