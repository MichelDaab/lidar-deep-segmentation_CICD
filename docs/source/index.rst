:github_url: https://github.com/IGNF/lidar-deep-segmentation

Lidar-Deep-Segmentation > Documentation
===================================================

.. include:: introduction.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/setup_install
   tutorials/prepare_dataset
   tutorials/make_predictions


.. toctree::
   :maxdepth: 1
   :caption: Guides

   how_to/add_new_data_signature
   how_to/train_new_model

.. toctree::
   :maxdepth: 1
   :caption: Background

   background/interpolation
   background/data_optimization

.. TODO: assure that all dosctrings are in third-personn mode.
.. TODO: find a way to document hydra config ; perhaps by switching to a full dataclasses mode.

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   apidoc/scripts
   apidoc/configs
   apidoc/lidar_multiclass.data
   apidoc/lidar_multiclass.model
   apidoc/lidar_multiclass.models.modules
   apidoc/lidar_multiclass.callbacks
   apidoc/lidar_multiclass.utils


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`