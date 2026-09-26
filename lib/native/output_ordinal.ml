include
  Core.Tagged_int.Make
    (struct
      let prefix = ""
    end)
    ()

let zero = of_int 0
let one = of_int 1
let indexed xs = List.mapi (fun i x -> (of_int i, x)) xs
