import pickle
import pydarn
from matplotlib.dates import date2num
import datetime as dt
import pandas as pd
import json
import numpy as np
import datetime as dt

def get_scan_nums(scan_array):
    """
    Группирует данные в сканы, используя поле scan из FITACF.
    Новый скан начинается при изменении значения scan.
    Костыль - но пока фиг с ним: так нулевые лучи всегда отделяться от остальных будут. В рантайме склеиваются пока пускай
    """
    if len(scan_array) == 0:
        return np.array([], dtype=int)
    
    scan_nums = np.zeros(len(scan_array), dtype=int)
    current_scan = 0
    
    for i in range(1, len(scan_array)):
        if scan_array[i] != scan_array[i-1]:  # Значение scan изменилось -> новый скан
            current_scan += 1
        scan_nums[i] = current_scan
    
    # Первая запись всегда относится к скану 0 (scan_nums[0] уже 0)
    return scan_nums

def get_datestr(year, month, day):
    return '%d-%02d-%02d' % (year, month, day)

def convert_db_from_dict(data_dict, date, rad, data_dir='./data', pickl=False):
    year, month, day = date[0], date[1], date[2]
    date_str = get_datestr(year, month, day)

    # Декодируем JSON строки обратно в массивы
    gate = np.hstack([json.loads(x) for x in data_dict['gate']]).astype(float)
    vel = np.hstack([json.loads(x) for x in data_dict['velocity']])
    wid = np.hstack([json.loads(x) for x in data_dict['width']])
    power = np.hstack([json.loads(x) for x in data_dict['power']])
    phi0 = np.hstack([json.loads(x) for x in data_dict['phi0']])
    elev = np.hstack([json.loads(x) for x in data_dict['elevation']])
    trad_gs_flg = np.hstack([json.loads(x) for x in data_dict['gsflg']])
    
    time, beam, freq, nsky, nsch, scan_flat = [], [], [], [], [], []
    num_scatter = data_dict['num_scatter']
    
    for i in range(len(num_scatter)):
        time.extend(date2num([data_dict['datetime'][i]] * num_scatter[i]))
        beam.extend([float(data_dict['beam'][i])] * num_scatter[i])
        freq.extend([float(data_dict['frequency'][i])] * num_scatter[i])
        nsky.extend([float(data_dict['nsky'][i])] * num_scatter[i])
        nsch.extend([float(data_dict['nsch'][i])] * num_scatter[i])
        scan_flat.extend([float(data_dict['scan'][i])] * num_scatter[i])  #
    
    time = np.array(time)
    beam = np.array(beam)
    freq = np.array(freq)
    nsky = np.array(nsky)
    nsch = np.array(nsch)
    scan_flat = np.array(scan_flat)  #

    nbeam = np.max(beam) + 1 if len(beam) > 0 else 0
    nrang = data_dict['nrang'][0] if data_dict['nrang'] else 0
    scan_nums = get_scan_nums(scan_flat)  # Сканы ищет не по времени, а по проходу
    scan_nums = scan_nums // 2
    
    # ===== ДИАГНОСТИКА =====
    # После формирования scan_nums
    scan_lengths = [np.sum(scan_nums == s) for s in np.unique(scan_nums)]
    print(f"Найдено сканов: {len(scan_lengths)}")
    print(f"Длины сканов (количество точек): {scan_lengths[:20]}")
    print(f"Уникальные лучи в первых 5 сканах:")
    for s in range(min(5, len(np.unique(scan_nums)))):
        beams_in_scan = beam[scan_nums == s]
        print(f"  Скан {s}: лучи {np.unique(beams_in_scan)}")
    # ========================

    for s in np.unique(scan_nums):
        mask = scan_nums == s
        # Сортируем по beam внутри скана
        sort_idx = np.argsort(beam[mask])
        gate[mask] = gate[mask][sort_idx]
        beam[mask] = beam[mask][sort_idx]
        vel[mask] = vel[mask][sort_idx]
        wid[mask] = wid[mask][sort_idx]
        time[mask] = time[mask][sort_idx]
        trad_gs_flg[mask] = trad_gs_flg[mask][sort_idx]
        elev[mask] = elev[mask][sort_idx]

    gate_scans = []
    beam_scans = []
    vel_scans = []
    wid_scans = []
    time_scans = []
    trad_gs_flg_scans = []
    elv_scans = []

    for s in np.unique(scan_nums):
        scan_mask = scan_nums == s
        gate_scans.append(gate[scan_mask])
        beam_scans.append(beam[scan_mask])
        vel_scans.append(vel[scan_mask])
        wid_scans.append(wid[scan_mask])
        time_scans.append(time[scan_mask])
        trad_gs_flg_scans.append(trad_gs_flg[scan_mask])
        elv_scans.append(elev[scan_mask])

    data = {
        'gate': gate_scans, 
        'beam': beam_scans, 
        'vel': vel_scans, 
        'wid': wid_scans,
        'time': time_scans, 
        'trad_gsflg': trad_gs_flg_scans, 
        'elv': elv_scans,
        'nrang': nrang, 
        'nbeam': nbeam
    }
    
    filename = "%s/%s_%s_scans" % (data_dir, rad, date_str)


    print("Уникальных времён:", len(np.unique(np.hstack(data['time']))))
    print("Всего точек:", sum(len(t) for t in data['time']))
    print("Уникальных лучей:", np.unique(np.hstack(data['beam'])))

    if pickl:
        pickle.dump(data, open(filename + ".pickle", 'wb'))
        print(f"Saved pickle: {filename}.pickle")
    else:
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename + ".csv")
        print(f"Saved CSV: {filename}.csv")
    
    return True
    
def gather_fitacf(fitacf_data, start_time, end_time, beam='*'):
    data_dict = {
        'datetime': [],
        'beam': [],
        'nrang': [],
        'num_scatter': [],
        'frequency': [],
        'nsky': [],
        'nsch': [],
        'power': [],
        'velocity': [],
        'width': [],
        'gate': [],
        'gsflg': [],
        'hop': [],
        'elevation': [],
        'phi0': [],
        'scan': []
    }

    for record in fitacf_data:
        # Проверяем наличие обязательных полей
        required_fields = ['time.yr', 'time.mo', 'time.dy', 'time.hr', 'time.mt', 
                          'time.sc', 'time.us', 'bmnum', 'nrang', 'tfreq']
        
        # Пропускаем записи без обязательных полей
        if not all(field in record for field in required_fields):
            continue
            
        # Пропускаем записи без данных (нет slist или пустой slist)
        if 'slist' not in record or record['slist'] is None or len(record['slist']) == 0:
            continue

        # Создаем datetime объект
        try:
            time_obj = dt.datetime(
                record['time.yr'], record['time.mo'], record['time.dy'],
                record['time.hr'], record['time.mt'], record['time.sc'],
                record['time.us']
            )
        except:
            continue

        # Фильтрация по времени
        if not (start_time <= time_obj <= end_time):
            continue

        # Фильтрация по лучу
        current_beam = record['bmnum']
        if beam != '*' and current_beam != beam:
            continue

        # Заполняем data_dict
        data_dict['datetime'].append(time_obj)
        data_dict['beam'].append(current_beam)
        data_dict['nrang'].append(record['nrang'])
        data_dict['frequency'].append(record['tfreq'] * 0.001)  # kHz to MHz
        
        # Количество scatter точек
        slist = record['slist']
        if isinstance(slist, (list, np.ndarray)) and len(slist) > 0:
            num_scatter = len(slist)
        else:
            continue  # Пропускаем если нет scatter точек
            
        data_dict['num_scatter'].append(num_scatter)
        
        # Шумовые параметры (с значениями по умолчанию)
        data_dict['nsky'].append(record.get('noise.sky', 0))
        data_dict['nsch'].append(record.get('noise.search', record.get('noise.mean', 0)))
        
        # Массивные параметры - преобразуем в списки и сериализуем в JSON
        data_dict['power'].append(json.dumps(_safe_to_list(record.get('p_l', []))))
        data_dict['velocity'].append(json.dumps(_safe_to_list(record.get('v', []))))
        data_dict['width'].append(json.dumps(_safe_to_list(record.get('w_l', []))))
        data_dict['gate'].append(json.dumps(_safe_to_list(slist)))
        data_dict['gsflg'].append(json.dumps(_safe_to_list(record.get('gflg', []))))
        data_dict['hop'].append(json.dumps([]))  # Заглушка для hop
        data_dict['elevation'].append(json.dumps(_safe_to_list(record.get('elv', []))))
        data_dict['phi0'].append(json.dumps(_safe_to_list(record.get('phi0', []))))

        data_dict['scan'].append(record.get('scan', 0))  #

    if not data_dict['datetime']:
        print("Warning: После фильтрации не найдено данных!")
        return False

    print(f"Успешно обработано {len(data_dict['datetime'])} записей")
    return data_dict

def _safe_to_list(data):
    if data is None:
        return []
    if isinstance(data, (list, np.ndarray)):
        return [float(x) if np.isscalar(x) else x for x in data]
    return [float(data)] if np.isscalar(data) else [data]

from glob import glob
import bz2 

if __name__ == "__main__":
    rad = 'cve'
    # date = (2021, 2, 1)  # год, месяц, день
    start_date = dt.datetime.strptime('20210201', '%Y%m%d')
    start_time = start_date.replace(hour=0, minute=0, second=0)
    end_date = dt.datetime.strptime('20210203', '%Y%m%d')
    end_time = end_date.replace(hour=23, minute=59, second=59)
    current_date = start_date
    while current_date <= end_date:
        print(f"Данные за {current_date.strftime('%Y%m%d')}:")
        fitacf_files = glob(f"./poli-data/{current_date.strftime('%Y%m%d')}.*.fitacf.bz2")
        print(f"\tНайдено {len(fitacf_files)} файлов.")
        if len(fitacf_files) == 0:
            print("\tДанных нет. Пропускаем...")
            current_date += dt.timedelta(days=1)
            continue
        fitacf_data = []
        fitacf_files.sort()
        print("\tЧитаем fitacf-файлы...")
        for fitacf_file in fitacf_files:
            with bz2.open(fitacf_file) as fp:
                fitacf_stream = fp.read()
        
            reader = pydarn.SuperDARNRead(fitacf_stream, True)
            records = reader.read_fitacf()
            fitacf_data += records
        print("\tОбрабатываем данные в формат pickle...")
        day_start = current_date.replace(hour=0, minute=0, second=0)
        day_end = current_date.replace(hour=23, minute=59, second=59)
        data_dict = gather_fitacf(fitacf_data, day_start, day_end)
        date_tuple = (current_date.year, current_date.month, current_date.day)
        convert_db_from_dict(data_dict, date_tuple, rad, pickl=True)

        # Проверка структуры
        import pickle
        test_pickle = f"./data/{rad}_{date_tuple[0]}-{date_tuple[1]:02d}-{date_tuple[2]:02d}_scans.pickle"
        with open(test_pickle, 'rb') as f:
            test_data = pickle.load(f)
        print(f"Number of scans: {len(test_data['time'])}")
        print(f"First scan: {test_data['time'][0][:5] if len(test_data['time'][0]) > 0 else 'empty'}")

        print("\tГотово!\n")
        current_date += dt.timedelta(days=1)