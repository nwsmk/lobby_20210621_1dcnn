import os
import numpy as np


def dbinv(x):
    out = np.power(10, x / 10)
    return out


def get_total_rss(csientry):
    rssi_mag = 0
    if csientry["rssi_a"] != 0:
        rssi_mag = rssi_mag + dbinv(csientry["rssi_a"])
    if csientry["rssi_b"] != 0:
        rssi_mag = rssi_mag + dbinv(csientry["rssi_b"])
    if csientry["rssi_c"] != 0:
        rssi_mag = rssi_mag + dbinv(csientry["rssi_c"])

    out = (10 * np.log10(rssi_mag)) - 44 - csientry["agc"]
    return out


def get_scaled_csi(csientry):
    csi = csientry["csi"]

    # calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = csi * np.conjugate(csi)
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csientry))
    scale = rssi_pwr / (csi_pwr / 30)

    if csientry["noise"] == -127:
        noise_db = -92
    else:
        noise_db = csientry["noise"]

    thermal_noise_pwr = dbinv(noise_db)

    # quantization error
    quant_error_pwr = scale * (csientry["Nrx"] * csientry["Ntx"])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    ret = csi * np.sqrt(scale / total_noise_pwr)

    if csientry["Ntx"] == 2:
        ret = ret * np.sqrt(2)
    elif csientry["Ntx"] == 3:
        ret = ret * np.sqrt(dbinv(4.5))

    return ret


# read bfree (ported from .c source file)
def read_bfree(inbytes):
    timestamp_low = np.uint32(inbytes[0] + (inbytes[1] << 8) + (inbytes[2] << 16) + (inbytes[3] << 24))
    bfree_count = np.uint16(inbytes[4] + (inbytes[5] << 8))
    Nrx = np.uint16(inbytes[8])
    Ntx = np.uint16(inbytes[9])
    rssi_a = np.uint16(inbytes[10])
    rssi_b = np.uint16(inbytes[11])
    rssi_c = np.uint16(inbytes[12])
    noise = np.int8(inbytes[13])
    agc = np.uint16(inbytes[14])
    antenna_sel = np.uint16(inbytes[15])
    datalen = np.uint16(inbytes[16] + (inbytes[17] << 8))
    fake_rate_n_flags = np.uint16(inbytes[18] + (inbytes[19] << 8))
    calc_len = np.uint16((30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8)
    payload = np.uint8(inbytes[20:(20 + datalen)])

    if datalen == calc_len:
        # calculate csi
        index = 0
        csi = np.zeros((30, Ntx, Nrx), dtype=np.complex_)
        for i in range(30):
            index = index + 3
            remainder = index % 8
            for m in range(Ntx):
                for n in range(Nrx):
                    tmp_re = np.int8(payload[np.int(index / 8)] >> remainder | payload[np.int(index / 8) + 1]
                                     << (8 - remainder))
                    tmp_csi_re = tmp_re
                    tmp_im = np.int8(payload[np.int(index / 8) + 1] >> remainder | payload[np.int(index / 8) + 2]
                                     << (8 - remainder))
                    tmp_csi_im = tmp_im
                    csi[i][m, n] = tmp_csi_re + (1j * tmp_csi_im)
                    index = index + 16

        # calculate permutation array
        perm = np.zeros((3,), dtype=np.int8)
        perm[0] = (antenna_sel & 0x3) + 1
        perm[1] = ((antenna_sel >> 2) & 0x3) + 1
        perm[2] = ((antenna_sel >> 4) & 0x3) + 1

        # create output dict
        outcell = {
            "timestamp_low": timestamp_low,
            "bfee_count": bfree_count,
            "Nrx": Nrx,
            "Ntx": Ntx,
            "rssi_a": rssi_a,
            "rssi_b": rssi_b,
            "rssi_c": rssi_c,
            "noise": noise,
            "agc": agc,
            "perm": perm,
            "rate": fake_rate_n_flags,
            "csi": csi
        }

    return outcell


# read the beamforming feedback file
def read_bf_file(fname):
    # low-level file opening
    with open(fname, "rb") as f:

        # move to the end of the file
        f.seek(0, os.SEEK_END)
        # length of the data
        datalen = f.tell()

        # move to the beginning of the file
        f.seek(0, os.SEEK_SET)

        # output variables
        ret = []
        cur = 0
        count = 0
        broken_perm = 0
        triangle = [1, 3, 6]

        while cur < datalen - 3:
            field_len = np.fromfile(f, dtype=np.dtype('>u2'), count=1)[0]
            code = np.fromfile(f, dtype=np.dtype('>u1'), count=1)[0]
            cur = cur + 3

            if code == 187:
                databytes = np.fromfile(f, dtype=np.dtype('>u1'), count=field_len - 1)
                cur = cur + field_len - 1
            else:
                f.seek(field_len - 1, os.SEEK_CUR)
                cur = cur + field_len - 1

            if code == 187:
                count = count + 1
                tmpret = read_bfree(databytes)

                perm = tmpret["perm"]
                Nrx = tmpret["Nrx"]

                if Nrx != 1:
                    if np.sum(perm) != triangle[Nrx - 1]:
                        if broken_perm == 0:
                            broken_perm = 1
                            print("found CSI with invalid perm")
                    else:
                        newindex = perm[:Nrx] - 1
                        oldindex = np.arange(Nrx)
                        oldcsi = tmpret["csi"]
                        newcsi = np.zeros(oldcsi.shape, dtype=np.complex_)
                        for i in range(30):
                            newcsi[i][:, newindex] = oldcsi[i][:, oldindex]
                        tmpret["csi"] = newcsi
                        ret.append(tmpret)

        outret = ret[:count]

    return outret
