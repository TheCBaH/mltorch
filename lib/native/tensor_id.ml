include
  Core.Tagged_int.Make
    (struct
      let prefix = "t"
    end)
    ()

let jsont = Jsont.map ~dec:of_int ~enc:to_int Jsont.int
