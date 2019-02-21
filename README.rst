=============================
 Dream Market CAPTCHA Solver
=============================

Copyright (C) 2019 Monolith Works Inc. All rights reserved.
Licensed under the terms of GNU General Public License Version 3 or later.

This project solves CAPTCHA on Dream Market.

Grabber
=======

.. code-block:: shell

    $ mkdir dm
    $ (cd dm && python ../grab.py)
    targetting url: http://k3pd243s57fttnpa.onion
    getting session id
    got session id: ui4c4ltmd5r1svhqh0akf96q22 (7.11 sec.)
    0: fetching
    ...

Trainer
=======

.. code-block:: shell

    $ python ./dmguess/train.py -o cnn.h5 -a dm.answers.json dm/

Guesser
=======

.. code-block:: shell

    $ python ./dmguess/guess.py -m cnn.h5 /path/to/images/
