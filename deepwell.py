# -*- coding: utf-8 -*-
from numpy import *
import random
import ctypes
import time
import sys
import os

softmax_en = 0x0
CMP_EN     = True
pdll = ctypes.cdll.LoadLibrary('cypress.so')
map_dict = {0:0x01, 1:0x02, 2:0x04, 3:0x08, 4:0x10, 5:0x20, 6:0x40, 7:0x80, 8:0x100, 9:0x200}

def RESET ():
    sc_wdata = 0x0000003F
    pdll.write_cypress_32bit(sc_wdata)
def WR_PARA (data_i,op_code):
    sc_wdata = (data_i << 2) | op_code
    pdll.write_cypress_32bit(sc_wdata)
def RD_PARA (rd_addr):
    op_code = 0x23
    sc_wdata = (rd_addr << 8) | op_code
    pdll.write_cypress_32bit(sc_wdata)
def CONF_H (h_st_addr,h_size,op_code):
    sc_wdata = ( h_size << 17) | ( h_st_addr << 6) | op_code
    pdll.write_cypress_32bit(sc_wdata)
    sc_wdata = ( h_size << 19) | ( h_st_addr << 8) | op_code  
def CONF_RW_WHT (rw_st_addr,rw_size,op_code):
    sc_wdata = (rw_st_addr << 17) | (rw_size << 6) | op_code
    pdll.write_cypress_32bit(sc_wdata)     
def START_TRAIN (nef_cs):
    nef_cs = nef_cs & 0x3
    train_st = 0
    sc_wdata = train_st | nef_cs
    pdll.write_cypress_32bit(sc_wdata)
def START_TEST (nef_cs):
    nef_cs = nef_cs & 0x3
    test_st = 1 << 2
    sc_wdata = test_st | nef_cs
    pdll.write_cypress_32bit(sc_wdata) 
def WR_REG (reg,reg_addr,nef_cs):
    nef_cs = nef_cs & 0x3
    reg_addr = reg_addr & 0xF
    sc_wdata = (reg << 8) | (reg_addr << 2 ) | nef_cs
    pdll.write_cypress_32bit(sc_wdata)  
    #print "%x ,"%sc_wdata
def CONF_BUS (rw_size,op_code,nef_cs):
    nef_cs = nef_cs & 0x3
    op_code = op_code & 0xF
    sc_wdata = (rw_size << 6) | (op_code << 2 ) | nef_cs
    pdll.write_cypress_32bit(sc_wdata)  
def WD_EN (op_code):
    nef_cs = 0
    reg = op_code << 1
    reg_addr = 0x7
    WR_REG(reg,reg_addr,nef_cs)
def RD_WHT(mem_sel,op_code):
    sc_wdata = (mem_sel << 6) | op_code
    pdll.write_cypress_32bit(sc_wdata)   

def CFG_PLL (pll_io_reg,pll_core_reg1,pll_core_reg0):
    nef_cs=0x0
    reg_addr=0x6
    pll_io_reg=pll_io_reg&0x7f
    
    pll_core_reg0=pll_core_reg0&0x7
    pll_core_reg1=pll_core_reg1&0x7
    
    pll_io_en=0x0
    pll_core_en=0x0
    pll_io_reg_t=(pll_io_reg << 1) | pll_io_en
    pll_core_reg=(pll_core_reg1 << 4) | (pll_core_reg0 << 1) | pll_core_en    
    reg=(pll_io_reg_t << 8) | pll_core_reg
    WR_REG(reg,reg_addr,nef_cs)

    pll_io_en=0x1
    pll_core_en=0x1
    pll_io_reg_t=(pll_io_reg << 1) | pll_io_en
    pll_core_reg=(pll_core_reg1 << 4) | (pll_core_reg0 << 1) | pll_core_en    
    reg=(pll_io_reg_t << 8) | pll_core_reg
    WR_REG(reg,reg_addr,nef_cs)

    pll_io_en=0x0
    pll_core_en=0x0
    pll_io_reg_t=(pll_io_reg << 1) | pll_io_en
    pll_core_reg=(pll_core_reg1 << 4) | (pll_core_reg0 << 1) | pll_core_en    
    reg=(pll_io_reg_t << 8) | pll_core_reg
    WR_REG(reg,reg_addr,nef_cs)

def INIT():
    pdll.cypress_init()
    pll_io_reg=11   #11,14,19,29
    pll_core_reg0=4 #3,4,3,4
    pll_core_reg1=1 #1,1,2,3
    clkio=2400/(2*(pll_io_reg+1))
    clkcore=2400/(pll_core_reg1*pll_core_reg0)
    CFG_PLL (pll_io_reg,pll_core_reg1,pll_core_reg0)

def wait_for_idle(nef_cs):
    rd_com = []
    c_state = 0
    while c_state != 1 :     
        reg = 18
        reg_addr = 0xD
        WR_REG(reg,reg_addr,nef_cs)
        rdb = []
        buf = (ctypes.c_ubyte * (1*4))(*rdb)
        count = pdll.read_cypress(buf,  1*4)
        c_state = buf[0]>>2
        #print "-Read core %d c_state = %d"%(nef_cs,c_state)
    #print "-out wait for idle"

def Read_Weight(N_test,nef_cs): 
    wait_for_idle(nef_cs)
    DMA_wr_size = N_test*4
    op_code = 0xE #dma read
    CONF_BUS (DMA_wr_size,op_code,nef_cs)

    read_result = []
    rdb = []
    buf = (ctypes.c_ubyte * (N_test*16))(*rdb)
    count = pdll.read_cypress(buf,  N_test*16)
    for ii in range(0, count):
        read_result.append(buf[ii])

    return read_result

def read_weights(weight_128b_size,mem_sel,nef_cs):
    wait_for_idle(nef_cs)
#1 config rw_st_addr, rw_size
    rw_size = weight_128b_size
    rw_st_addr = 0x0
    reg = (rw_st_addr << 11) | rw_size
    reg_addr = 0x4
    WR_REG(reg,reg_addr,nef_cs)
#2 star Readweigt, mem_sel
    # mem select and start write weights
    reg = mem_sel
    reg_addr = 0x3
    WR_REG(reg,reg_addr,nef_cs)

    if rw_size > 1024:
        print "Readweight number is %d x128bit, which is error, can not more than 1024 !"%rw_size
    read_result = []
    loop_num = rw_size/32
    if (rw_size%32 > 0) :
        loop_num = loop_num +1
    for i in range(loop_num):
        if i == loop_num -1:
            read_result = read_result + Read_Weight(int(rw_size-i*32),nef_cs)
        else:
            read_result = read_result + Read_Weight(int(32),nef_cs)

    return read_result

def read_lable(N_test, nef_cs):
    read_result = []
    rdb = []
    buf = (ctypes.c_ubyte * (N_test*16))(*rdb)
    DMA_wr_size = N_test*4
    op_code = 0xE #dma read 
    CONF_BUS (DMA_wr_size,op_code,nef_cs)
    count = pdll.read_cypress(buf,  N_test*16)
    for ii in range(0, count, 16):#一个128bit结果只有最低位有效
        read_result.append(buf[ii])
    return read_result

def read_vector(N_test, nef_cs):
    read_result = []
    rdb = []
    buf = (ctypes.c_float * (N_test*3*4))(*rdb)#一个结果是12个32位浮点数
    DMA_wr_size = N_test*3*4
    op_code = 0xE #dma read 
    CONF_BUS (DMA_wr_size,op_code,nef_cs)
    count = pdll.read_sfmax_vector_cypress(buf)
    for ii in range(0, N_test*3*4):
        read_result.append(buf[ii])
    return read_result
    
def Train(data, label, ls, ds, ss, hs, clear_M_en, nef_cs = 0):
    global softmax_en, CMP_EN
    N_train     = data.shape[0]
    size        = data.shape[1] * N_train
    DMA_wr_size = size/4
    ptn_size    = data.shape[1]/16
    if size%1024 > 0 and size%1024 < 28:
        print "write size must be greater than 27Byte\r\n"
    nef_st_addr = 0x000
    reg = (hs << 11) | nef_st_addr
    reg_addr = 0x5
    WR_REG(reg,reg_addr,nef_cs)

    seed = 0x20080820
    seed_l= seed & 0xFFFF
    reg = seed_l
    reg_addr = 0x8
    WR_REG(reg,reg_addr,nef_cs)

    seed_h= seed >> 16
    reg = seed_h
    reg_addr = 0x9
    WR_REG(reg,reg_addr,nef_cs)

    reg = (ss << 7)|(ls << 3)| ds
    reg_addr = 0xa
    WR_REG(reg,reg_addr,nef_cs)

    #训练阶段不能开任何输出
    softmax_out_en = 0x0
    softmax_cmp_en = 0x0

    Y_out_en = 0x0
    cmp_rlt_en = 0x0
    reg = (ptn_size << 6)|(cmp_rlt_en << 5)|(Y_out_en << 4)| (softmax_cmp_en << 3)|(softmax_out_en << 2)|(softmax_en << 1)| clear_M_en
    reg_addr = 0xb
    WR_REG(reg,reg_addr,nef_cs)
    ptn_data = []
    for q in range (N_train):                
        ptn_data_once = []
        img   = data[q,:]
        index = label[q,:].argmax()
        int_label = int(index)           
        tmp_data = ( int(img[1]) << 14 ) | ( int_label << 10 ) | map_dict[int_label]
        for i in range (0, 2):
            tmp = ( tmp_data >> i*8 ) & 0xFF
            ptn_data_once.append(tmp)
        for j in range (2, len(img)):
            tmp_int = int(img[j])
            ptn_data_once.append(tmp_int)
        ptn_data.extend(ptn_data_once)

    buf = (ctypes.c_ubyte * size)(*ptn_data)
    reg = N_train
    reg_addr = 0x0
    WR_REG(reg,reg_addr,nef_cs)
    op_code = 0xF
    CONF_BUS (DMA_wr_size,op_code,nef_cs)
    pdll.write_cypress_16k_block(buf, size)

def Test(data, ls, ds, ss, hs, nef_cs = 0):
    global softmax_en, CMP_EN
    N_test      = data.shape[0]
    size        = data.shape[1] * N_test
    DMA_wr_size = size/4
    ptn_size    = data.shape[1]/16

    if size%1024 > 0 and size%1024 < 28:
        print "write size must be greater than 27Byte\r\n"

    nef_st_addr = 0x000
    reg = (hs << 11) | nef_st_addr
    reg_addr = 0x5
    WR_REG(reg,reg_addr,nef_cs)

    reg = (ss << 7)|(ls << 3)| ds
    reg_addr = 0xa
    WR_REG(reg,reg_addr,nef_cs)
    
    clear_M_en = 0x0
    #位宽为128bit
    if CMP_EN:#使能cmp en,输出结果是一个lable，返回为一个位宽的数据，既128bit，但只有低4bit有效
        softmax_out_en = 0x0
        softmax_cmp_en = 0x1
    else:#输出结果是12个小数(3个128bit，按照自定义格式的组合(softmax结果))
        softmax_out_en = 0x1
        softmax_cmp_en = 0x0

    Y_out_en = 0x0
    cmp_rlt_en = 0x0

    if softmax_en == 0:
        softmax_out_en = 0x0
        softmax_cmp_en = 0x0
        if CMP_EN:
            cmp_rlt_en = 0x1
        else:
            Y_out_en   = 0x1

    reg = (ptn_size << 6)|(cmp_rlt_en << 5)|(Y_out_en << 4)| (softmax_cmp_en << 3)|(softmax_out_en << 2)|(softmax_en << 1)| clear_M_en
    reg_addr = 0xb
    WR_REG(reg,reg_addr,nef_cs)

    int_label  = 0
    ptn_data = []
    for q in range(N_test):                
        ptn_data_once = []
        img   = data[q,:]        
        tmp_data = ( int(img[1]) << 14 ) | ( int_label << 10 ) | map_dict[int_label]
        for i in range (0, 2):
            tmp = ( tmp_data >> i*8 ) & 0xFF
            ptn_data_once.append(tmp)
        for j in range (2, len(img)):
            tmp_int = int(img[j])
            ptn_data_once.append(tmp_int)
        ptn_data.extend(ptn_data_once)

    buf = (ctypes.c_ubyte * size)(*ptn_data)

    reg = N_test
    reg_addr = 0x1
    WR_REG(reg, reg_addr, nef_cs)
    op_code = 0xF
    CONF_BUS(DMA_wr_size, op_code, nef_cs)
    pdll.write_cypress_16k_block(buf, size)
    wait_for_idle(nef_cs)
    if CMP_EN:
        return read_lable(N_test,0)
    else:
        return read_vector(N_test,0)

def clear_weight():
    clear_M_en = 1
    array_data = zeros([1, 3776])
    array_vec  = zeros([1, 10])
    Train(array_data, array_vec, 0, 0, 0, 0x7ff, clear_M_en)
