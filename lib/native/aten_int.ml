module Dim =
  Core.Tagged_int.Make
    (struct
      let prefix = ""
    end)
    ()

module Index =
  Core.Tagged_int.Make
    (struct
      let prefix = ""
    end)
    ()

module Size =
  Core.Tagged_int.Make
    (struct
      let prefix = ""
    end)
    ()

module Step =
  Core.Tagged_int.Make
    (struct
      let prefix = ""
    end)
    ()
