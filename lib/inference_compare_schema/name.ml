(* Names the case or graph gives to external operands. *)

module Constant =
  Text_value.Make_identifier
    (struct
      let role = "constant_name"
    end)
    ()

module Explicit_state =
  Text_value.Make_identifier
    (struct
      let role = "state_name"
    end)
    ()

module Scope_component =
  Text_value.Make_identifier
    (struct
      let role = "scope_component"
    end)
    ()

module User_input =
  Text_value.Make_identifier
    (struct
      let role = "user_input_name"
    end)
    ()
