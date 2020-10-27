import sys
import numpy as np
import netCDF4 as nc
import datetime

for i in sys.argv[1:]:
    d = nc.Dataset(i)
    tas = d['tas'][:]
    times = d['time'][:]
    units = d['time'].units
    units = units.split('since ')[1]
    date_split = units.split('-')
    init_date = datetime.datetime(int(date_split[0]), int(date_split[1]),
                                  int(date_split[2].split('T')[0]), int(0),
                                  int(0))
    _times = [str(init_date + datetime.timedelta(hours=int(j))) for j in times]
    print _times[0]
    np.savez_compressed(i.split('.nc')[0] + '.npz', tas=np.array(tas), times=_times)
