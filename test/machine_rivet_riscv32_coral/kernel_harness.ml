(* A generated function called directly on the VME core: its slots are the
   caller's inputs (read-only, with [pad] bytes of [fill] after each so that a
   read past an operand's end is a defined one) and outputs (zeroed by the
   runtime), its address table is [a0], and the
   function receives [args] in [a0]-[a7], a slot's address or an immediate,
   and returns. *)

module A = Machine_rivet_riscv32.Rv32_asm
module L = Coral_layout

let dir = Sys.getenv "CORALNPU_DATA"
let configuration = "vme-core-mini-axi"
let entry = "kernel_entry"

let caller ~symbol ~args =
  A.module_ "kernel_caller" (fun () ->
      let i = A.insn and r = A.reg in
      [ A.text; A.align 4; A.global entry; A.label entry ]
      @ [
          i "addi" [ r "sp"; r "sp"; A.imm (-16) ];
          i "sw" [ r "ra"; A.mem "sp" 12 ];
          i "mv" [ r "t0"; r "a0" ];
        ]
      @ List.concat
          (List.mapi
             (fun k arg ->
               let reg = r (Printf.sprintf "a%d" k) in
               match arg with
               | `Slot j -> [ i "lw" [ reg; A.mem "t0" (4 * j) ] ]
               | `Imm n -> [ i "li" [ reg; A.imm n ] ])
             args)
      @ [
          i "call" [ A.sym symbol ];
          i "lw" [ r "ra"; A.mem "sp" 12 ];
          i "addi" [ r "sp"; r "sp"; A.imm 16 ];
          i "li" [ r "a0"; A.imm 0 ];
          i "ret" [];
        ])

type result = {
  outputs : string list;  (** each output slot's bytes, in order *)
  cycles : int64;
  retired : int64;
  segments : (string * int) list;
      (** each segment's bytes, zero fill included *)
}

(* [inputs] are the bytes of each read-only slot; [outputs] the sizes of the
   written ones. The slots are bound in that order; [args] defaults to all. *)
let run ?(sram = false) ?(pad = 0) ?(fill = '\000') ?args
    ?(cycles = 200_000_000) ?(configuration = configuration) ~symbol ~modules
    ~inputs ~outputs () =
  let ( let* ) = Result.bind in
  let args =
    Option.value args
      ~default:
        (List.init
           (List.length inputs + List.length outputs)
           (fun k -> `Slot k))
  in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let* call = caller ~symbol ~args in
  let slot placement size =
    {
      L.Slot.align = 16L;
      home = (if sram then L.Home.Ram else L.Home.Any);
      placement;
      size = Int64.of_int size;
    }
  in
  let padded s = s ^ String.make pad fill in
  let slots =
    List.map (fun s -> slot L.Placement.In (String.length s + pad)) inputs
    @ List.map (fun n -> slot L.Placement.Out n) outputs
  in
  let* image =
    Result.map_error
      (Fmt.str "%a" Coral_image.pp_error)
      (Err.payload
         (Coral_image.build
            ~rv32_profile:
              Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
            ~matrix:true profile ~entry ~kernel:(call :: modules) slots))
  in
  let n_in = List.length inputs in
  let* o =
    Result.map_error
      (Fmt.str "%a" Coral_run.Run_error.pp)
      (Err.payload
         (Coral_run.run
            ~flags:[ Printf.sprintf "--cycles=%d" cycles ]
            ~dir image
            ~bound:(fun i ->
              if i < n_in then Some (padded (List.nth inputs i)) else None)))
  in
  Ok
    {
      outputs =
        List.mapi (fun k _ -> List.assoc (n_in + k) o.Coral_run.outputs) outputs;
      cycles = o.Coral_run.mcycle;
      retired = o.Coral_run.minstret;
      segments =
        List.map
          (fun (g : Image.segment) ->
            (g.Image.name, String.length g.Image.bytes + g.Image.zero_fill))
          image.Coral_image.image.Image.segments;
    }

let f32_bytes xs =
  let b = Bytes.create (4 * Array.length xs) in
  Array.iteri
    (fun i x -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x))
    xs;
  Bytes.to_string b

let f32_array s =
  Array.init
    (String.length s / 4)
    (fun i -> Int32.float_of_bits (String.get_int32_le s (4 * i)))

let bits32 s i = String.get_int32_le s (4 * i)

(* Two binary32 values equal as bits, every NaN equal to every NaN. *)
let same_f32 a b =
  (Float.is_nan a && Float.is_nan b)
  || Int32.bits_of_float a = Int32.bits_of_float b
