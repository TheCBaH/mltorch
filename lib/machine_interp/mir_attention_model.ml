(* The interpreters' model of an attention helper ({!Mir_attention}): the
   reference evaluator over the cells the pointer arguments address, Q, K and V
   bfloat16 (the high half of a binary32), the mask and O binary32. A cell
   outside its object or never written is a defect of the program, as it would
   be for the generated code. *)

open Machine_ir

let model (h : Mir_helper.t) =
  Option.map
    (fun (d : Attention_block.t) ->
      {
        Mir_helper_model.name = h.Mir_helper.name;
        version = h.Mir_helper.version;
        run =
          (fun memory args ->
            let q, k, v, o, mask =
              match (args, d.Attention_block.mask) with
              | ( [
                    Mir_datum.Ptr q;
                    Mir_datum.Ptr k;
                    Mir_datum.Ptr v;
                    Mir_datum.Ptr o;
                  ],
                  None ) ->
                  (q, k, v, o, None)
              | ( [
                    Mir_datum.Ptr q;
                    Mir_datum.Ptr k;
                    Mir_datum.Ptr v;
                    Mir_datum.Ptr o;
                    Mir_datum.Ptr m;
                  ],
                  Some _ ) ->
                  (q, k, v, o, Some m)
              | _ ->
                  invalid_arg
                    "Mir_attention_model: an argument the descriptor's \
                     signature rejects"
            in
            let at ~bytes p off =
              match Mir_memory.offset_by p (Int64.mul bytes off) with
              | Some q -> q
              | None -> invalid_arg "Mir_attention_model: an element outside"
            in
            let read ~bytes p off =
              match
                Mir_memory.load memory (at ~bytes p off) ~bytes ~align:bytes
              with
              | Ok bits ->
                  Int32.float_of_bits
                    (Int64.to_int32
                       (Int64.shift_left bits (8 * (4 - Int64.to_int bytes))))
              | Error _ ->
                  invalid_arg "Mir_attention_model: an unreadable element"
            in
            let eb =
              match d.Attention_block.precision with
              | Attention_block.Precision.Bf16_paired -> 2L
              | Attention_block.Precision.F32_fused
              | Attention_block.Precision.F32_ordered ->
                  4L
            in
            Attention_block.eval d ~q:(read ~bytes:eb q) ~k:(read ~bytes:eb k)
              ~v:(read ~bytes:eb v)
              ~mask:
                (match mask with
                | Some m -> read ~bytes:4L m
                | None -> fun _ -> 0.)
              ~write:(fun off x ->
                match
                  Mir_memory.store memory (at ~bytes:4L o off) ~bytes:4L
                    ~align:4L
                    (Int64.logand
                       (Int64.of_int32 (Int32.bits_of_float x))
                       0xFFFF_FFFFL)
                with
                | Ok () -> ()
                | Error _ ->
                    invalid_arg "Mir_attention_model: an unwritable element");
            Mir_helper_model.Returns []);
      })
    (Mir_attention.of_helper h)

(* The models a program's attention helpers need. *)
let models (helpers : Mir_helper.t list) = List.filter_map model helpers
