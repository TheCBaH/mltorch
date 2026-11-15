module Param =
  Core.Tagged_int.Make
    (struct
      let prefix = "param"
    end)
    ()

module Site =
  Core.Tagged_int.Make
    (struct
      let prefix = "site"
    end)
    ()

module Name =
  Core.Tagged_int.Make
    (struct
      let prefix = "name"
    end)
    ()
