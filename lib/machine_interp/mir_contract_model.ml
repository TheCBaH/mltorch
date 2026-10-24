(* The interpreters' model of a contraction helper ({!Mir_contract}): the
   reference evaluator over the cells the three pointer arguments address, A
   and B binary32 or bfloat16 (its bits the high half of a binary32), C
   binary32. A cell outside its object or never written is a defect of the
   program, as it would be for the generated code. *)

open Machine_ir

let model (h : Mir_helper.t) =
  Option.map
    (fun d ->
      {
        Mir_helper_model.name = h.Mir_helper.name;
        version = h.Mir_helper.version;
        run =
          (fun memory -> function
            | [ Mir_datum.Ptr a; Mir_datum.Ptr b; Mir_datum.Ptr c ] ->
                let at ~bytes p off =
                  match Mir_memory.offset_by p (Int64.mul bytes off) with
                  | Some q -> q
                  | None -> invalid_arg "Mir_contract_model: an element outside"
                in
                let read ~bytes p off =
                  match
                    Mir_memory.load memory (at ~bytes p off) ~bytes ~align:bytes
                  with
                  | Ok bits ->
                      Int32.float_of_bits
                        (Int64.to_int32
                           (Int64.shift_left bits
                              (8 * (4 - Int64.to_int bytes))))
                  | Error _ ->
                      invalid_arg "Mir_contract_model: an unreadable element"
                in
                let operand =
                  read ~bytes:(Contraction.Element.bytes d.Contraction.format)
                in
                Contraction.eval d ~a:(operand a) ~b:(operand b)
                  ~c:(read ~bytes:4L c) ~write:(fun off x ->
                    match
                      Mir_memory.store memory (at ~bytes:4L c off) ~bytes:4L
                        ~align:4L
                        (Int64.logand
                           (Int64.of_int32 (Int32.bits_of_float x))
                           0xFFFF_FFFFL)
                    with
                    | Ok () -> ()
                    | Error _ ->
                        invalid_arg "Mir_contract_model: an unwritable element");
                Mir_helper_model.Returns []
            | _ ->
                invalid_arg
                  "Mir_contract_model: an argument the descriptor's signature \
                   rejects");
      })
    (Mir_contract.of_helper h)

(* The models a program's contraction helpers need. *)
let models (helpers : Mir_helper.t list) = List.filter_map model helpers
