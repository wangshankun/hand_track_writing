#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include "cyusb.h"

static int timeout = 1;
static cyusb_handle *h1 = NULL;

void print_hex_str(const void* buf , unsigned int size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int textLength = lineLength;
    unsigned int ix = 0 ;
    unsigned int jx = 0 ;

    for (ix = 0 ; ix < size ; ix += lineLength) {
        sprintf(line, "%.8xh: ", ix);
// 打印16进制
        for (jx = 0 ; jx != lineLength ; jx++) {
            if (ix + jx >= size) {
                sprintf(line + (11 + jx * 3), "   "); // 处理最后一行空白
                if (ix + jx == size)
                    textLength = jx;  // 处理最后一行文本截断
            } else
                sprintf(line + (11 + jx * 3), "%.2X ", * (str + ix + jx));
        }
// 打印字符串
        {
            memcpy(text, str + ix, lineLength);
            pc = text;
            while (pc != text + lineLength) {
                if ((unsigned char)*pc < 0x20) // 空格之前为控制码
                    *pc = '.';                 // 控制码转成'.'显示
                pc++;
            }
            text[textLength] = '\0';
            sprintf(line + (11 + lineLength * 3), "; %s", text);
        }

        printf("%s\n", line);
    }
}

extern "C"{

int write_cypress_32bit(unsigned int cmd)
{
    int r;
    int transferred = 0;
    unsigned int cmd_r[7] = {cmd, cmd, cmd, cmd, cmd, cmd, cmd};
    r = cyusb_bulk_transfer(h1, 0x01, (unsigned char*)&cmd_r, 4 * 7, &transferred, timeout * 1000);
    if ( r != 0 )
    {
       printf("cypress_write_32bit error\r\n");
       return -1;
    }

    return transferred;
}

//每次读至少16字节的整数倍
int read_cypress(unsigned char* buf_r, int size)
{
    if (size%4 != 0)
    {
        printf("read size must be 16Byte multiple\r\n");
        return -2;
    }

    int r;
    int transferred = 0;

    r = cyusb_bulk_transfer(h1, 0x81, buf_r, size, &transferred, timeout * 1000);
    if ( r != 0 ) 
    {
       printf("read_cypress error\r\n");
       return -1;
    }
    return transferred;
}

//softmax的结果默认为48字节
//每次读至少16字节的整数倍,一个结果size恒定为48，为12个浮点型数组
int read_sfmax_vector_cypress(float* buf_r, int n_test)
{
    int size  = 48 * n_test;
    if (size%4 != 0)
    {
        printf("read size must be 16Byte multiple\r\n");
        return -2;
    }
    int i, r;
    int transferred = 0;
    unsigned char* tmp_buf = (unsigned char*)malloc(size);
    r = cyusb_bulk_transfer(h1, 0x81, tmp_buf, size, &transferred, timeout * 1000);
    if ( r != 0 )
    {
       printf("read_sfmax_vector_cypress error\r\n");
       free(tmp_buf);
       return -1;
    }
    flag* arry =  (flag*)tmp_buf;
    //按照自定义(0-10bit表示小数)的数据类型，转为通用的float32
    for (i = 0; i< size/4; i++)
    {
        buf_r[i] = arry[i].f10 * (float)1/2 + arry[i].f9 * (float)1/4 +  arry[i].f8 * (float)1/8 +  arry[i].f7 * (float)1/16 + \
                    arry[i].f6 * (float)1/32 + arry[i].f5 * (float)1/64 + arry[i].f4 * (float)1/128 + arry[i].f3 * (float)1/256 + \
                     arry[i].f2 * (float)1/512 + arry[i].f1 * (float)1/1024 + arry[i].f0 * (float)1/2048;
    }
    free(tmp_buf);
    return transferred;
}
//每次写至少28个字节
int write_cypress_16k_block(unsigned char* buf_w, int size)
{
    //printf("DMA write 16kByte per package. \r\n");
    if ((size%1024 > 0)&&(size%1024 < 28))
    {
        printf("write size must be greater than 27Byte\r\n");
        return -3;
    }

    int i,r,offset;
    int transferred = 0;
    int tmp_transed = 0;
    int cyc = size/16384;
    int re  = size%16384;
    for(i = 0,offset = 0; i < cyc; i++, offset += 16384)
    {
        tmp_transed = 0;
        r = cyusb_bulk_transfer(h1, 0x01, buf_w + offset, 16384, &tmp_transed, timeout * 1000);
        if ( r != 0 ) 
        {
           printf("write_cypress_16k_block cyc data error\r\n");
           return -1;
        }
        transferred = transferred + tmp_transed;
        
    }
    //printf("offset:%d  re:%d\r\n",offset,re);
    tmp_transed = 0;
    r = cyusb_bulk_transfer(h1, 0x01, buf_w + offset, re, &tmp_transed, timeout * 1000);
    if ( r != 0 ) 
    {
        printf("write_cypress_16k_block re data error\r\n");
        return -1;
    }
    transferred = transferred + tmp_transed;
    
    int zero_package;
    zero_package = re%1024;
    tmp_transed = 0;
    if ( zero_package == 0) //when last package is 1KB, send 0 package to triger send
    {
        r = cyusb_bulk_transfer(h1, 0x01, buf_w + offset, 0, &tmp_transed, timeout * 1000);
        if ( r != 0 ) 
        {
            printf("write 0 package data data error\r\n");
            return -1;
        }
    }
    return transferred;
}


int write_cypress(unsigned char* buf_w, int size)
{
    if ((size%1024 > 0)&&(size%1024 < 28))
    {
        printf("write size must be greater than 27Byte\r\n");
        return -3;
    }

    int r;
    int transferred = 0;

    r = cyusb_bulk_transfer(h1, 0x01, buf_w, size, &transferred, timeout * 1000);
    if ( r != 0 )
    {
       printf("write_cypress data error\r\n");
       return -1;
    }

    return transferred;
}


void cypress_exit(void)
{
    cyusb_close();
}


int cypress_init(void)
{
    int r;
    r = cyusb_open();
    if ( r < 0 )
    {
        printf("Error opening library\n");
        return -126;
    }
    else if ( r == 0 )
    {
        printf("No device found\n");
        return -127;
    }
    if ( r > 1 ) 
    {
        printf("More than 1 devices of interest found. Disconnect unwanted devices\n");
        return -126;
    }
    h1 = cyusb_gethandle(0);
    if ( cyusb_getvendor(h1) != 0x04b4 ) 
    {
        printf("Cypress chipset not detected\n");
        cyusb_close();
        return -125;
    }
    r = cyusb_kernel_driver_active(h1, 0);
    if ( r != 0 )
    {
       printf("kernel driver active. Exitting\n");
       cyusb_close();
       return -124;
    }
    r = cyusb_claim_interface(h1, 0);
    if ( r != 0 )
    {
       printf("Error in claiming interface\n");
       cyusb_close();
       return -123;
    }

    return 0;
}
}
