# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v1.0.1].

## Version [1.1.7] - 2025-06-30

### Added

- Added helper function that returns feature names after preprocessing.

### Changed

- Helper function that returns the feature names for tabular datasets no longer returns "target" name.

## Version [1.1.6] - 2025-06-30

### Added

- Helper function that returns the feature names for tabular datasets.

## Version [1.1.5] - 2025-03-12

### Changed

- Fixed a small bug in `nfinal`. 

## Version [1.1.4] - 2025-03-11

### Changed

- For tabular datasets, it is now possible to return training and test data, ensuring no data leakage from the transformations involved in preprocessing. [#34]

## Version [1.1.3] - 2025-03-11

### Changed

- Ensure that *Adult* data is properly pre-processed (i.e. categorical variables are one-hot encoded). [#30]

### Added

- Helper function `format_header!(df::DataFrame)` to apply some standard formatting to column names. [#30]
- Added option `return_cats::Bool` where applicable to allow users to retrieve the indices of categorical features. [#30]

## Version [1.1.2] - 2025-01-10

### Changed

- Improved and streamlined some assertions related to dataset sizes. [#29]

## Version [1.1.1] - 2025-01-10

### Changed

- Improved seeding behaviour for tabular and vision datasets. [#28]

## Version [1.1.0] - 2025-01-09

### Changed

- Changed the way the default seed is set to avoid overriding the global seed. [#26], [#27]
