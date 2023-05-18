# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import PerFed


def main():
    args = args_parser()
    perFed = PerFed(args)
    perFed.server()
    perFed.global_test()


if __name__ == '__main__':
    main()
