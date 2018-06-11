import numpy
import btk
import os
import math
import re
import gc
from progress.bar import Bar
from random import random

###########################################################################
#                           read data functions                           #
###########################################################################

def get_marker_names(_filename):
    _reader = btk.btkAcquisitionFileReader() # build a btk _reader object
    _reader.SetFilename(_filename) # set a filename to the _reader
    _reader.Update()
    _acq = _reader.GetOutput()
    _frame_max =  _acq.GetPointFrameNumber() #get the number of frames
    _all_label = []
    for _j_index in range(0, _frame_max):
      _points = {}
      for _i_index in range(_acq.GetPointNumber()):
        _point = _acq.GetPoint(_i_index)
        _label = _point.GetLabel()
        _all_label.append(_label)
    return _all_label
 
def get_positions(_filename):
    _reader = btk.btkAcquisitionFileReader() # build a btk _reader object
    _reader.SetFilename(_filename) # set a filename to the _reader
    _reader.Update()
    _acq = _reader.GetOutput()
    
    _frame_max =  _acq.GetPointFrameNumber() #get the number of frames

    _all_points = []
    for _j_index in range(0, _frame_max):
      _points = {}
      for _i_index in range(_acq.GetPointNumber()):
        _point = _acq.GetPoint(_i_index)
        _label = _point.GetLabel()
        _values = _point.GetValues() #returns a numpy array of the data, row=frame, col=coordinates
        _points[_label] = _values[_j_index][:]
        _all_points.append(_points)

    return _all_points
    
def get_domain_for_each_marker(_data):
  _domains = {}
  for _key in _data[0].keys():
    _min_values = [_data[0][_key][i] for i in range(0,3)]
    _max_values = [_data[0][_key][i] for i in range(0,3)]
    for _j_index in range(1, len(_data)):
      _values = _data[_j_index][_key]
      for _i_index in range(0,3):
        if _values[_i_index] < _min_values[_i_index]:
          _min_values[_i_index] = _values[_i_index]
        if _values[_i_index] > _max_values[_i_index]:
          _max_values[_i_index] = _values[_i_index]
    _domains[_key] = [_min_values, _max_values]
  return _domains

def walk(_dir, _pattern, _method):
    _dir = os.path.abspath(_dir)
    for _file in [_file for _file in os.listdir(_dir) if not _file in [".", "..", ".svn", ".git"]]:
        _nfile = os.path.join(_dir, _file)
        if _pattern.search(_nfile):
            _method(_nfile)
        if os.path.isdir(_nfile):
            walk(_nfile, _pattern, _method)

def get_domains_for_all_files(_directory):
    _pattern = re.compile(r".*RBOHand.*.trb$")
    _files = []
    walk(_directory, _pattern, _files.append)
    _domains = [get_domain_for_each_marker(get_positions(_filename)) for _filename in _files]
    _global_domains = {}
    _keys = _domains[0].keys()
    for _key in _keys:
        _domain = []
        for d in _domains:
            _minimum = d[_key][0]
            _maximum = d[_key][1]
            if _domain == []:
                _domain = [[_minimum[0], _minimum[1], _minimum[2]], [_maximum[0], _maximum[1], _maximum[2]]]

            _d_min = [_domain[0][i] if _minimum[i] > _domain[0][i] else _minimum[i] for i in range(0,3)]
            _d_max = [_domain[1][i] if _maximum[i] < _domain[1][i] else _maximum[i] for i in range(0,3)]
        _global_domains[_key] = [_d_min, _d_max]
    return _global_domains

def scale_data_for_each_marker(_data, _domains):
  _scaled_data = {}
  for _key in _domains.keys():
    _new_data = []
    _min_values = _domains[_key][0]
    _max_values = _domains[_key][1]
    for _j_index in range(0, len(_data)):
      _values = _data[_j_index][_key]
      # if _values[0] != nan:
      if max([abs((_max_values[_i_index] - _min_values[_i_index])) for _i_index in range(0,3)]) > 0.00001:
        _values = [(_values[_i_index] - _min_values[_i_index]) /
                   (_max_values[_i_index] - _min_values[_i_index]) for _i_index in range(0,3)]
      _new_data.append(_values)
    _scaled_data[_key] = _new_data
  return _scaled_data

###########################################################################
#                            binning functions                            #
###########################################################################

def bin_value(_v, _bins):
  return min(int(_v * _bins), _bins-1)

def bin_vector(_v, _bins):
  return [min(int(_v[i] * _bins), _bins-1) for i in range(0,3)]

def bin_scaled_data_for_each_marker(_data, _bins):
  _new_data = {}
  for _key in _data.keys():
    _new_data[_key] = [bin_vector(_v, _bins) for _v in _data[_key]]
  return _new_data

def combine_bin_vector(_v, _bins):
    return sum([_v[i] * pow(_bins,i) for i in range(0, len(_v))])

def unique_valued_list(_lst):
    _myset = set(_lst)
    return list(_myset)

def relabel_vector(_lst):
    _mylst = unique_valued_list(_lst)
    return [_mylst.index(_v) for _v in _lst]

def combine_bins_for_each_marker(_data, _bins):
    _new_data = {}
    for _key in _data.keys():
        _new_data[_key] = relabel_vector([combine_bin_vector(_v, _bins) for _v in _data[_key]])
    return _new_data

def combine_random_variables(_lst_of_lsts, _bins):
    return relabel_vector([combine_bin_vector([_v[i] for _v in _lst_of_lsts], _bins) for i in range(0, len(_lst_of_lsts[1]))])
    
###########################################################################
#                   calculating probabilities from data                   #
###########################################################################

def emperical_joint_distribution(_w_prime, _w, _a):
  _p = numpy.zeros((max(_w_prime)+1, max(_w)+1, max(_a)+1))

  _l = len(_w_prime)
  for index in range(0, _l):
    _p[_w_prime[index], _w[index], _a[index]] = _p[_w_prime[index], _w[index], _a[index]] + 1.0

  for _i_index in range(0, _p.shape[0]):
    for _j_index in range(0, _p.shape[1]):
      for _k_index in range(0, _p.shape[2]):
        _p[_i_index,_j_index,_k_index] = _p[_i_index,_j_index,_k_index] / float(_l)

  _s = sum(sum(sum(_p)))
  _p = _p / _s
  return _p

def calc_p_w_prime_given_w(_joint_distribution):
    _p_w_prime_w = _joint_distribution.sum(axis=2)
    _p_w         = _joint_distribution.sum(axis=(0,2))
    for _w_prime in range(0,_joint_distribution.shape[0]):
        for _w in range(0, _joint_distribution.shape[1]):
            _p_w_prime_w[_w_prime, _w] = _p_w_prime_w[_w_prime, _w] / _p_w[_w]
    return _p_w_prime_w

def calc_p_w_prime_given_a(_joint_distribution):
    _p_w_prime_a = _joint_distribution.sum(axis=1)
    _p_a         = _joint_distribution.sum(axis=(0,1))
    for _w_prime in range(0,_joint_distribution.shape[0]):
        for _a in range(0, _joint_distribution.shape[2]):
            if _p_w_prime_a[_w_prime, _a] != 0.0 and _p_a[_a] != 0.0:
                _p_w_prime_a[_w_prime, _a] = _p_w_prime_a[_w_prime, _a] / _p_a[_a]
    return _p_w_prime_a

def calc_p_w_prime_given_w_a(_joint_distribution):
    _p_w_a               = _joint_distribution.sum(axis=0)
    _p_w_prime_given_w_a = numpy.zeros(_joint_distribution.shape)
    for _w_prime in range(0, _joint_distribution.shape[0]):
        for _w in range(0, _joint_distribution.shape[1]):
            for _a in range(0, _joint_distribution.shape[2]):
                if _joint_distribution[_w_prime, _w, _a] != 0.0 and _p_w_a[_w,_a] != 0.0:
                    _p_w_prime_given_w_a[_w_prime, _w, _a] = _joint_distribution[_w_prime, _w, _a] / _p_w_a[_w,_a]
    return _p_w_prime_given_w_a

###########################################################################
#                           MC quantifications                            #
###########################################################################

def calculate_concept_one(_joint_distribution):
    _p_w_prime_given_w   = calc_p_w_prime_given_w(_joint_distribution)
    _p_w_prime_given_w_a = calc_p_w_prime_given_w_a(_joint_distribution)
    _r = 0
    for _w_prime in range(0, _joint_distribution.shape[0]):
        for _w in range(0, _joint_distribution.shape[1]):
            for _a in range(0, _joint_distribution.shape[2]):
                if _joint_distribution[_w_prime, _w, _a] != 0.0 and \
                   _p_w_prime_given_w[_w_prime, _w]      != 0.0 and \
                   _p_w_prime_given_w_a[_w_prime, _w, _a] != 0.0:
                    _r = _joint_distribution[_w_prime, _w, _a] * \
                         (math.log(_p_w_prime_given_w_a[_w_prime, _w, _a], 2) -
                          math.log(_p_w_prime_given_w[_w_prime, _w], 2))
    return _r

def calculate_concept_two(_joint_distribution):
    _p_w_prime_given_a   = calc_p_w_prime_given_a(_joint_distribution)
    _p_w_prime_given_w_a = calc_p_w_prime_given_w_a(_joint_distribution)
    _r = 0
    for _w_prime in range(0, _joint_distribution.shape[0]):
        for _w in range(0, _joint_distribution.shape[1]):
            for _a in range(0, _joint_distribution.shape[2]):
                if _joint_distribution[_w_prime, _w, _a] != 0.0 and _p_w_prime_given_a[_w_prime, _a] != 0.0 and _p_w_prime_given_w_a[_w_prime, _w, _a] != 0.0:
                    _r = _joint_distribution[_w_prime, _w, _a] * \
                         (math.log(_p_w_prime_given_w_a[_w_prime, _w, _a], 2) -
                          math.log(_p_w_prime_given_a[_w_prime, _a], 2))
    return _r

###########################################################################
#                 unique information based quantification                 #
###########################################################################

epsilon = 0.000001

def ml_eqeq(_value, _l): # matlab == operator
  return numpy.array([1 if _value == x else 0 for x in numpy.ravel(_l)])

def ml_lesser_than_zero(_l):
  return numpy.array([1 if _x < 0.0  else 0 for _x in numpy.ravel(_l)])

def ml_greater_than_zero(_l):
  return numpy.array([1 if _x > 0.0  else 0 for _x in numpy.ravel(_l)])

def ml_geq_zero(_l):
  return numpy.array([1 if _x >= 0.0 else 0 for _x in numpy.ravel(_l)])

def ml_extract(_l, _idx):
  _r = numpy.empty(0)
  for _index in range(0, len(_l)):
      if _idx[_index] == 1:
          _r = numpy.concatenate((_r, [_l[_index]]))
  return _r

def kronecker_product(_a, _b):
  _la = len(_a)
  _lb = len(_b)
  _c = numpy.zeros(_la * _lb)
  # bar = Bar('Kronecker product', max=_la)
  for _a_index in range(0, _la):
    for _b_index in range(0, _lb):
      _c[_a_index * _lb + _b_index] = _a[_a_index] * _b[_b_index]
    # bar.next()
  # bar.finish()
  return _c

def calculate_bases(_nr_of_world_states, _nr_of_action_states): # tested
  _r = []
  _x_range = range(_nr_of_world_states)
  _y_range = range(_nr_of_world_states)
  _z_range = range(_nr_of_action_states)
  bar = Bar('Calculate bases', max=(_nr_of_world_states * _nr_of_world_states * _nr_of_action_states))
  for _x in range(0, _nr_of_world_states):
    for _y in range(1, _nr_of_world_states):
      for _z in range(1, _nr_of_action_states):
        _yp    = 0
        _zp    = 0
        _xe    = ml_eqeq(_x,  _x_range)
        _ye    = ml_eqeq(_y,  _y_range)
        _ype   = ml_eqeq(_yp, _y_range)
        _ze    = ml_eqeq(_z,  _z_range)
        _zpe   = ml_eqeq(_zp, _z_range)


        if use_numpy_kron == True:
          _tmp    = numpy.kron(_ye,  _xe)
          _dxyz   = numpy.kron(_ze, _tmp)

          _tmp    = numpy.kron(_ype, _xe)
          _dxypzp = numpy.kron(_zpe, _tmp)

          _tmp    = numpy.kron(_ype, _xe)
          _dxypz  = numpy.kron(_ze, _tmp)

          _tmp    = numpy.kron(_ye,  _xe)
          _dxyzp  = numpy.kron(_zpe, _tmp)

        else:
          _tmp    = kronecker_product(_ye,  _xe)
          _dxyz   = kronecker_product(_ze, _tmp)

          _tmp    = kronecker_product(_ype, _xe)
          _dxypzp = kronecker_product(_zpe, _tmp)

          _tmp    = kronecker_product(_ype, _xe)
          _dxypz  = kronecker_product(_ze, _tmp)

          _tmp    = kronecker_product(_ye,  _xe)
          _dxyzp  = kronecker_product(_zpe, _tmp)

        _r.append(_dxyz + _dxypzp - _dxypz - _dxyzp)
        bar.next()

        # del _dxyz 
        # del _dxypzp
        # del _dxypz
        # del _dxyzp
        # gc.collect()
    bar.finish()
  return numpy.asarray(_r)

def sample_from_delta_p(_p, _resolution):
  # assert if shape[0] != shape[1]
  _nr_of_world_states = _p.shape[0]
  _nr_of_action_states = _p.shape[2]
  print "p.shape: " + str(_p.shape)
  print "nr of world states:  " + str(_nr_of_world_states)
  print "nr of action states: " + str(_nr_of_action_states)
  if _nr_of_world_states <= 1 or _nr_of_action_states <= 1:
    print "not enough world or action states to proceed"
    return []
  _ps = numpy.ravel(_p)
  _dimension_of_delta_p = _nr_of_world_states * (_nr_of_world_states - 1) * (_nr_of_action_states - 1)
  _lst = [_ps] # list of return values
  _s   = 0

  _d = calculate_bases(_nr_of_world_states, _nr_of_action_states)

  _d = numpy.asmatrix(_d)

  bar = Bar('Sampling from Delta_P', max=_resolution)
  for _ in range(0, _resolution):
    _a  = numpy.random.randn(_dimension_of_delta_p)
    _na = numpy.linalg.norm(_a)
    _a  = _a / _na
    _a  = numpy.asmatrix(_a)

    _b = _a * _d

    if numpy.count_nonzero(ml_eqeq(0, _b)) == 0:
      _v  = numpy.add(_ps, _s)
      _v  = -_v
      _v  = _v / _b
      _v  = numpy.ravel(_v)
      _vs = numpy.sign(_b)
      _vs = numpy.ravel(_vs)

      _vl = ml_lesser_than_zero(_vs)
      _vg = ml_greater_than_zero(_vs)

      _vvl = ml_extract(_v, _vl)
      _vvg = ml_extract(_v, _vg)

      _tmax = min(_vvl)
      _tmin = max(_vvg)

      _trand = numpy.random.rand(1)

      _to = _trand * (_tmax - _tmin) + _tmin
     
      _tob = _to * _a * _d
      _sd = _s + _tob
     
      _sdp = numpy.add(_sd, _ps)

      _gsdp = ml_geq_zero(_sdp)

      _pgsdp = numpy.prod(_gsdp)

      if _pgsdp >= 1.0:
        _s = _sd

        _sp = _s + _ps
        _lst.append(_sp)

    bar.next()
  bar.finish()
  rlst = [numpy.reshape(numpy.ravel(elem),(_nr_of_world_states, _nr_of_world_states, _nr_of_action_states))
          for elem in _lst]
  return rlst


def entropy(_distribution):
  return -sum([0 if _v == 0 else _v * math.log2(_v) for _v in numpy.ravel(_distribution)])

# MI(W';W|A) = H(W',A) + H(W,A) - H(W',W,A) - H(A)
# MI(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
def mi_xygz(_distribution):
  _xz = numpy.sum(_distribution, 1)
  _yz = numpy.sum(_distribution, 0)
  _z  = numpy.sum(_distribution, (0,1))
  return entropy(_xz) + entropy(_yz) - entropy(_distribution) - entropy(_z)

# MI(X;Z|Y) = H(X,Y) + H(Y,Z) - H(X,Y,Z) - H(Y)
# MI(W';A|W) = H(W',W) + H(W,A) - H(W',W,A) - H(W)
def mi_xzgy(_distribution):
  _xy = numpy.sum(_distribution, 2)
  _yz = numpy.sum(_distribution, 0)
  _y  = numpy.sum(_distribution, (0,2))
  return entropy(_xy) + entropy(_yz) - entropy(_distribution) - entropy(_y)

def mi_xy(_xy):
  _x  = numpy.sum(_xy, 0)
  _y  = numpy.sum(_xy, 1)
  _r = 0
  for _x_index in range(0, len(_x)):
    for _y_index in range(0, len(_y)):
      if _xy[_x_index][_y_index] > 0.0 and _x[_x_index] > 0.0 and _y[_y_index] > 0.0:
        _r = _r + _xy[_x_index][_y_index] * \
             (math.log2(_xy[_x_index][_y_index]) -
              math.log2(_x[_x_index] * math.log2(_y[_y_index])))
  return _r

# CoI(X;Y;Z) = MI(X;Y) - MI(X;Y|Z)
def coinformation(_distribution):
  _xy      = numpy.sum(_distribution, 2)
  _mi_xy   = mi_xy(_xy)
  _mi_xygz = mi_xygz(_distribution)
  return _mi_xy - _mi_xygz

def information_decomposition(_joint_distribution, _resolution):
  _samples = sample_from_delta_p(_joint_distribution, _resolution)
  _mi_xygz = [mi_xygz(_p) for _p in _samples]
  _mi_xzgy = [mi_xzgy(_p) for _p in _samples]
  _coid    = coinformation(_joint_distribution)
  _coi     = [coinformation(_p) for _p in _samples]

  _synergistic   = min([v - _coid for v in _coi])
  _uniquewprimew = min(_mi_xygz)
  _uniquewprimea = min(_mi_xzgy)
  
  return _synergistic, _uniquewprimea, _uniquewprimew

def synergistic(_joint_distribution, _resolution):
  _samples = sample_from_delta_p(_joint_distribution, _resolution)
  if len(_samples) == 0:
    return 0.0
  _coid    = coinformation(_joint_distribution)
  _coi     = [coinformation(_p) for _p in _samples]

  _synergistic   = min([v - _coid for v in _coi])
  
  return _synergistic

def uniquewprimew(_joint_distribution, _resolution):
  _samples = sample_from_delta_p(_joint_distribution, _resolution)
  if len(_samples) == 0:
    return 0.0
  _mi_xygz = [mi_xygz(_p) for _p in _samples]

  _uniquewprimew = min(_mi_xygz)
  
  return _uniquewprimew

def uniquewprimea(_joint_distribution, _resolution):
  print "getting samples from Delta_P"
  _samples = sample_from_delta_p(_joint_distribution, _resolution)
  if len(_samples) == 0:
    return 0.0
  print "calculating mutual informations"
  _mi_xzgy = [mi_xzgy(_p) for _p in _samples]

  _uniquewprimea = min(_mi_xzgy)
  
  return _uniquewprimea
    
###########################################################################
#                            analyse functions                            #
###########################################################################

def analyse_directory(_parent, _nr_of_bins, _functions):
    print "reading all files and looking for their domains"
    _domains     = get_domains_for_all_files(_parent)
    
    _binned_actions = None
    
    _pattern = re.compile(r".*RBOHand.*.trb$")
    _files = []
    walk(_parent, _pattern, _files.append)
    
    # print "Only using the first three files for test reasons."
    # _files = _files[4:5]
    
    _results = {}
    
    for _f in _files:
        print "reading file " + _f
        _data = get_positions(_f)
        print "scaling data"
        _scaled_data = scale_data_for_each_marker(_data, _domains)
        print "binning data"
        _binned_data = bin_scaled_data_for_each_marker(_scaled_data, _nr_of_bins)
        print "combining data for each marker"
        _combined_binned_data = combine_bins_for_each_marker(_binned_data, _nr_of_bins)
        print "combining data of all markers"
        _combined_binned_data = combine_random_variables([_combined_binned_data[_key] for _key in _combined_binned_data.keys()], _nr_of_bins)
        print "randomising action data"
        _binned_actions = [int(random() * _nr_of_bins) for _ in range(1,len(_combined_binned_data))]
        print "calculate joint distribution"
        _jd = emperical_joint_distribution( \
              _combined_binned_data[2:len(_combined_binned_data)],
              _combined_binned_data[1:len(_combined_binned_data)-1],
              _binned_actions)
        
        print "joint distribution shape: " + str(_jd.shape)
        _r = {}
        for _key in _functions.keys():
            print "using method: " + _key
            _r[_key] = _functions[_key](_jd)
            print "result: " + str(_r[_key])
        _results[_f] = _r
    print "done."
    return _results
 
def analyse_per_finger_and_thumb_directory(_parent, _nr_of_bins, _functions):
    print "reading all files and looking for their domains"
    _domains     = get_domains_for_all_files(_parent)
    
    _binned_actions = None
    
    _pattern = re.compile(r".*RBOHand.*.trb$")
    _files = []
    walk(_parent, _pattern, _files.append)
    
    print "Only using the first three files for test reasons."
    _files = _files[0:3]
    
    _results = {}
    
    for _f in _files:
        print "reading file " + _f
        _data = get_positions(_f)
        print "scaling data"
        _scaled_data = scale_data_for_each_marker(_data, _domains)
        print "binning data"
        _binned_data = bin_scaled_data_for_each_marker(_scaled_data, _nr_of_bins)
        print "combining data for each marker"
        _combined_binned_data = combine_bins_for_each_marker(_binned_data, _nr_of_bins)
        #_combined_binned_data = combine_random_variables([_combined_binned_data[_key] for _key in _combined_binned_data.keys()], _nr_of_bins)
        print "randomising action data"
        _binned_actions = [int(random() * _nr_of_bins) for _ in range(1,len(_combined_binned_data))]
        for _marker_key in _combined_binned_data.keys():
          print "calculate joint distribution for marker: " + _marker_key
          _jd = emperical_joint_distribution( \
                _combined_binned_data[_marker_key][2:len(_combined_binned_data)],
                _combined_binned_data[_marker_key][1:len(_combined_binned_data)-1],
                _binned_actions)
          
          _r = {}
          for _key in _functions.keys():
              print "using method: " + _key
              _r[_key] = _functions[_key](_jd)
          _results[_marker_key + " " + _f] = _r
    print "done."
    return _results
    
def fn_uniquewprimew(_joint_distribution):
    return uniquewprimea(_joint_distribution, resolution)

def prefix(_s, _l):
  _r = _s
  while len(_r) < _l:
    _r = " " + _r
  return _r

def print_and_save_results(_filename, _results, _functions):
  _l = 0
  for _key in _functions.keys():
    if len(_key) > _l:
      _l = len(_key)

  _fd = open(_filename, "w")
  for _key in _results.keys():
      print _key
      _fd.write(_key + "\n")
      for _k in _results[_key].keys():
        _s = "   " + prefix(_k, _l) + ": " + str(_results[_key][_k])
        print _s
        _fd.write(_s + "\n")
  _fd.close()
    
###########################################################################
#                         change parameters here                          #
###########################################################################

# use_numpy_kron = True
# bins           = 30
# resolution     = 1000
bins           = 100
directory      = os.environ["HOME"] + "/data/20141104-rbohand2/"
output         = "results.txt"

functions      = {"One"             : calculate_concept_one,
                  "Two"             : calculate_concept_two
                  }
                  #"Unique(W':W\\A)" : fn_uniquewprimew}

###########################################################################
#                     end of parametrisation section                      #
###########################################################################

# results = analyse_per_finger_and_thumb_directory(directory, bins, functions)
results = analyse_directory(directory, bins, functions)
print_and_save_results(output, results, functions)
