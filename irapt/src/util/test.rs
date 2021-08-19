use core::str::FromStr;

pub fn parse_csv<T: FromStr>(data: &'static [u8]) -> impl Iterator<Item = impl Iterator<Item = T> + ExactSizeIterator> {
    let mut csv_reader = csv::ReaderBuilder::new();
    csv_reader.has_headers(false);
    let expected = csv_reader.from_reader(data).into_records();
    let expected_rows = expected.map(|expected| {
        let expected = expected.unwrap();
        (0..expected.len()).map(move |index| {
            let expected_str = expected.get(index).unwrap();
            expected_str.parse::<T>().unwrap_or_else(|_| panic!("unparsable csv test data"))
        })
    });
    expected_rows
}

/// Asserts that two expressions evaluating to iterators yielding float values, or structs containing float values,
/// are approximately equal to each other.
///
/// The third argument to the macro is either:
///
///  * A numeric literal, indicating absolute comparison should be performed, i.e. the two numbers should be within
///    the given distance of each other.
///
///  * A numeric literal followed by `%`, indicating relative comparison should be performed, i.e. the smaller value
///    should be within the given percentage of the larger value.
///
/// If the values yielded from the iterators are structs, the fields to compare are specified as the fourth
/// argument, preceded by dots and separated by spaces.
///
/// On panic, this macro will print the values of the mismatching float values and the index they were yielded at
/// from their iterators.
///
/// Like [`assert!`], a custom panic message can be provided as the last arguments.
///
/// # Examples
///
/// ```
/// assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2);
/// assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%);
/// assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%, "custom message with an {}", "argument");
///
/// struct Foo {
///     x: f64,
///     y: f64,
/// }
/// let actual = [Foo { x: 100.0, y: 181.0 }];
/// let expected = [Foo { x: 91.0, y: 200.0 }];
/// assert_iter_approx_eq!(&actual, &expected, 10%, .x .y);
/// ```
macro_rules! assert_iter_approx_eq {
    // absolute comparison within given distance
    ($actual:expr, $expected:expr, $max_dist:literal $(, $($arg:tt)*)?) => ({
        __assert_iter_approx_eq_impl!(
            $actual,
            $expected,
            max_dist => $max_dist,
            (_actual, _expected) => max_dist
                $(, $($arg)*)?
        );
    });

    // relative comparison within given percentage of larger number
    ($actual:expr, $expected:expr, $max_percentage_dist:literal % $(, $($arg:tt)*)?) => ({
        __assert_iter_approx_eq_impl!(
            $actual,
            $expected,
            max_percentage_dist => f64::from($max_percentage_dist) / 100.0,
            (actual, expected) => f64::from(actual.clone()).abs().max(f64::from(expected.clone()).abs()) * max_percentage_dist
                $(, $($arg)*)?
        );
    });
}

macro_rules! __assert_iter_approx_eq_impl {
    // plain comparison between two values
    (
        $actual:expr,
        $expected:expr,
        $max_dist_ident:ident => $max_dist:expr,
        ($actual_item_ident:ident, $expected_item_ident:ident) => $item_max_dist:expr
            $(, $($arg:tt),* $(,)?)?
    ) => ({
        __assert_iter_approx_eq_impl_2!(
            actual => $actual,
            expected => $expected,
            $max_dist_ident => $max_dist,
            format => ($($($arg),*)?),
            index => __assert_approx_eq_impl!(
                actual.clone() => format_args!(concat!(stringify!($actual), "[{}]"), index),
                expected.clone() => format_args!(concat!(stringify!($expected), "[{}]"), index),
                {
                    let ($actual_item_ident, $expected_item_ident) = (actual, expected);
                    $item_max_dist
                },
                "{}",
                format,
            )
        )
    });

    // comparison between fields of a struct
    (
        $actual:expr,
        $expected:expr,
        $max_dist_ident:ident => $max_dist:expr,
        ($actual_item_ident:ident, $expected_item_ident:ident) => $item_max_dist:expr,
        $(.$field:ident)+ $(, $($arg:tt),* $(,)?)?
    ) => ({
        __assert_iter_approx_eq_impl_2!(
            actual => $actual,
            expected => $expected,
            $max_dist_ident => $max_dist,
            format => ($($($arg),*)?),
            index => { $(__assert_approx_eq_impl!(
                actual.$field => format_args!(concat!(stringify!($actual), "[{}].", stringify!($field)), index),
                expected.$field => format_args!(concat!(stringify!($expected), "[{}].", stringify!($field)), index),
                {
                    let ($actual_item_ident, $expected_item_ident) = (actual.$field, expected.$field);
                    $item_max_dist
                },
                "{}",
                format,
            ); )+ }
        )
    });
}

macro_rules! __assert_iter_approx_eq_impl_2 {
    ($actual_ident:ident => $actual:expr, $expected_ident:ident => $expected:expr, $max_dist_ident:ident => $max_dist:expr,
     $format_ident:ident => (), $index_ident:ident => $code:expr) => (
        __assert_iter_approx_eq_impl_2!($actual_ident => $actual, $expected_ident => $expected, $max_dist_ident => $max_dist,
                                        $format_ident => (""), $index_ident => $code)
    );
    ($actual_ident:ident => $actual:expr, $expected_ident:ident => $expected:expr, $max_dist_ident:ident => $max_dist:expr, $format_ident:ident => ($($arg:tt),+),
     $index_ident:ident => $code:expr) => ({
         use itertools::zip_eq;
         let $max_dist_ident = $max_dist;
         zip_eq($actual, $expected).enumerate().for_each(|($index_ident, ($actual_ident, $expected_ident))| {
             match format_args!($($arg),+) {
                 $format_ident => {
                     $code
                 }
             }
         });
     });
}

macro_rules! __assert_approx_eq_impl {
    ($actual:expr => $actual_fmt:expr, $expected:expr => $expected_fmt:expr, $max_dist:expr, $($arg:tt),+ $(,)?) => ({
        let dist = f64::from($actual - $expected).abs();
        if dist > $max_dist {
            assert_eq!(
                $actual,
                $expected,
                "abs({} - {}) = {} > {}: {}",
                $actual_fmt,
                $expected_fmt,
                dist,
                $max_dist,
                format_args!($($arg),+),
            );
        }
    })
}

#[test]
fn test_assert_iter_approx_eq() {
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2);
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2,);
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2, "");
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2, "",);
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2, "{}", "");
    assert_iter_approx_eq!(&[0.010f64, 0.020f64], &[0.011f64, 0.021f64], 1e-2, "{}", "",);

    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%);
    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%,);
    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%, "");
    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%, "",);
    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%, "{}", "");
    assert_iter_approx_eq!(&[100f64, 181f64], &[91f64, 200f64], 10%, "{}", "",);

    struct Foo {
        x: f64,
        y: f64,
    }
    let actual = [Foo { x: 100.0, y: 181.0 }];
    let expected = [Foo { x: 91.0, y: 200.0 }];
    assert_iter_approx_eq!(&actual, &expected, 10%, .x);
    assert_iter_approx_eq!(&actual, &expected, 10%, .x,);
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y);
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y,);
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y, "");
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y, "",);
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y, "{}", "");
    assert_iter_approx_eq!(&actual, &expected, 10%, .x .y, "{}", "",);
}

#[test]
#[should_panic]
fn test_assert_iter_approx_eq_absolute_fail() {
    assert_iter_approx_eq!(&[100.0f64], &[100.2f64], 1e-1);
}

#[test]
#[should_panic]
fn test_assert_iter_approx_eq_relative_fail() {
    assert_iter_approx_eq!(&[100f64], &[121f64], 10%);
}
