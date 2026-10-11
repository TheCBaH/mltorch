(* Explanatory text. Neither module is an identifier or an executable policy. *)

module Description =
  Text_value.Make_prose
    (struct
      let role = "description"
    end)
    ()

module Diagnostic_text =
  Text_value.Make_prose
    (struct
      let role = "diagnostic_text"
    end)
    ()
