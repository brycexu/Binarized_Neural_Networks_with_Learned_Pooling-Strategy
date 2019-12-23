#!/usr/bin/env python3

import os
from pprint import pprint

if __name__ == "__main__":
    pprint({k:os.environ[k] for k in os.environ})