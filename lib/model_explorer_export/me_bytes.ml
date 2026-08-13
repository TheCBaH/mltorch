let two_to_53 = 9_007_199_254_740_992.

let checked_length n ~max =
  if
    Float.is_nan n || Float.is_infinite n
    || n <> Float.floor n
    || n < 0. || n >= two_to_53
  then Err.fail (`Bad_length n)
  else
    let n64 = Int64.of_float n in
    if Int64.compare n64 max > 0 then Err.fail (`Over_limit n64)
    else if Int64.compare n64 Me_limits.Hard.jsoo_safe_bytes > 0 then
      Err.fail (`Over_limit n64)
    else Ok (Int64.to_int n64)
