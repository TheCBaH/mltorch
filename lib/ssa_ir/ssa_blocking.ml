module Rows =
  Core.Tagged_int.Make
    (struct
      let prefix = "rows"
    end)
    ()

module Unroll =
  Core.Tagged_int.Make
    (struct
      let prefix = "unroll"
    end)
    ()

module Group =
  Core.Tagged_int.Make
    (struct
      let prefix = "group"
    end)
    ()

module Parts =
  Core.Tagged_int.Make
    (struct
      let prefix = "parts"
    end)
    ()
