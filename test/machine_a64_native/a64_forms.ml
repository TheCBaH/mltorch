(* The AArch64 form instances the conformance harness probes: each admitted
   form with its immediates, conditions and sizes enumerated, its assembly
   template over named operands, and its selected-IR counterpart. *)

open Machine_ir
open Machine_target_aarch64
open A64_op

type reg = G of Sz.t | F of Fsz.t

module Form = struct
  type t = {
    name : string;
    inputs : reg list;
    result : reg option;
    asm : string;
        (** the instruction(s), operands named %[d] %[a] %[b] %[c] %[base] *)
    mem : bool;
    make :
      Mir_value.t list ->
      Mir_value.t ->
      [ `Op of A64_op.t | `Test of A64_op.test ];
        (** the form over virtual inputs and the seeded condition value *)
  }
end

let reg_ref name = function
  | G Sz.X -> Printf.sprintf "%%x[%s]" name
  | G Sz.W -> Printf.sprintf "%%w[%s]" name
  | F Fsz.D -> Printf.sprintf "%%d[%s]" name
  | F Fsz.S -> Printf.sprintf "%%s[%s]" name

let ty_of = function
  | G Sz.X -> Mir_type.i64
  | G Sz.W -> Mir_type.i32
  | F Fsz.D -> Mir_type.F64
  | F Fsz.S -> Mir_type.F32

let names = [ "a"; "b"; "c" ]

(* A three-operand form [ins d, a, b, ...] over registers of [r]. *)
let form ?(result = true) ?inputs name r asm make =
  let inputs = Option.value inputs ~default:[ r; r ] in
  let args = List.mapi (fun i k -> reg_ref (List.nth names i) k) inputs in
  let d = if result then [ reg_ref "d" r ] else [] in
  {
    Form.name;
    inputs;
    result = (if result then Some r else None);
    asm = Printf.sprintf "%s %s" asm (String.concat ", " (d @ args));
    mem = false;
    make =
      (fun vs f ->
        ignore f;
        `Op (make vs));
  }

let sz_name = function Sz.X -> "x" | Sz.W -> "w"
let fsz_name = function Fsz.D -> "d" | Fsz.S -> "s"
let conds = Cond.[ Eq; Ge; Gt; Hi; Hs; Le; Lo; Ls; Lt; Mi; Ne; Pl; Vc; Vs ]

let forms =
  let szs = [ Sz.W; Sz.X ] and fszs = [ Fsz.S; Fsz.D ] in
  let arg vs k = List.nth vs k in
  List.concat
    [
      List.concat_map
        (fun sz ->
          let g = G sz and n = sz_name sz in
          [
            form ("add." ^ n) g "add" (fun vs -> Add (sz, arg vs 0, arg vs 1));
            form ("sub." ^ n) g "sub" (fun vs -> Sub (sz, arg vs 0, arg vs 1));
            form ("mul." ^ n) g "mul" (fun vs -> Mul (sz, arg vs 0, arg vs 1));
            form ("sdiv." ^ n) g "sdiv" (fun vs ->
                Sdiv (sz, arg vs 0, arg vs 1));
            form ~inputs:[ g; g; g ] ("msub." ^ n) g "msub" (fun vs ->
                Msub (sz, arg vs 0, arg vs 1, arg vs 2));
            form ("and." ^ n) g "and" (fun vs ->
                Logic (Logic.And, sz, arg vs 0, arg vs 1));
            form ("eor." ^ n) g "eor" (fun vs ->
                Logic (Logic.Eor, sz, arg vs 0, arg vs 1));
            form ("orr." ^ n) g "orr" (fun vs ->
                Logic (Logic.Orr, sz, arg vs 0, arg vs 1));
            form ~inputs:[ g ] ("mov." ^ n) g "mov" (fun vs ->
                Mov (sz, arg vs 0));
            {
              (form ~result:false ("cmp." ^ n) g "cmp" (fun vs ->
                   Cmp (sz, arg vs 0, arg vs 1)))
              with
              Form.result = None;
            };
          ]
          @ List.map
              (fun k ->
                {
                  (form ~inputs:[ g ] (Printf.sprintf "add.%s#%Ld" n k) g "add"
                     (fun vs -> Add_imm (sz, arg vs 0, k)))
                  with
                  Form.asm =
                    Printf.sprintf "add %s, %s, #%Ld" (reg_ref "d" g)
                      (reg_ref "a" g) k;
                })
              [ 0L; 1L; 4095L; 4096L; 16773120L ]
          @ List.map
              (fun k ->
                {
                  (form ~inputs:[ g ] (Printf.sprintf "sub.%s#%Ld" n k) g "sub"
                     (fun vs -> Sub_imm (sz, arg vs 0, k)))
                  with
                  Form.asm =
                    Printf.sprintf "sub %s, %s, #%Ld" (reg_ref "d" g)
                      (reg_ref "a" g) k;
                })
              [ 0L; 1L; 4095L; 4096L; 16773120L ]
          @ List.map
              (fun k ->
                {
                  (form ~result:false ~inputs:[ g ]
                     (Printf.sprintf "cmp.%s#%Ld" n k) g "cmp" (fun vs ->
                       Cmp_imm (sz, arg vs 0, k)))
                  with
                  Form.asm = Printf.sprintf "cmp %s, #%Ld" (reg_ref "a" g) k;
                })
              [ 0L; 1L; 4095L ]
          @ List.concat_map
              (fun (o, on) ->
                List.map
                  (fun k ->
                    {
                      (form ~inputs:[ g ] (Printf.sprintf "%s.%s#0x%Lx" on n k)
                         g on (fun vs -> Logic_imm (o, sz, arg vs 0, k)))
                      with
                      Form.asm =
                        Printf.sprintf "%s %s, %s, #0x%Lx" on (reg_ref "d" g)
                          (reg_ref "a" g) k;
                    })
                  (match sz with
                  | Sz.W -> [ 1L; 0xFFL; 0x5555_5555L; 0xFFFF_FFFEL ]
                  | Sz.X ->
                      [ 1L; 0xFF00_FF00_FF00_FF00L; 0x7FFF_FFFF_FFFF_FFFFL ]))
              [ (Logic.And, "and"); (Logic.Eor, "eor"); (Logic.Orr, "orr") ]
          @ List.concat_map
              (fun (o, on) ->
                List.map
                  (fun k ->
                    {
                      (form ~inputs:[ g ] (Printf.sprintf "%s.%s#%d" on n k)
                         g on (fun vs -> Shift_imm (o, sz, arg vs 0, k)))
                      with
                      Form.asm =
                        Printf.sprintf "%s %s, %s, #%d" on (reg_ref "d" g)
                          (reg_ref "a" g) k;
                    })
                  [ 0; 1; 13; (match sz with Sz.W -> 31 | Sz.X -> 63) ])
              [ (Shift.Asr, "asr"); (Shift.Lsl, "lsl"); (Shift.Lsr, "lsr") ]
          @ List.concat_map
              (fun c ->
                [
                  {
                    (form
                       ("csel." ^ n ^ "." ^ Cond.name c)
                       g "csel"
                       (fun vs ->
                         Csel
                           ( sz,
                             c,
                             Mir_value.
                               {
                                 id = Mir_id.Value.of_int 9;
                                 ty = Mir_type.Flags;
                               },
                             arg vs 0,
                             arg vs 1 )))
                    with
                    Form.asm =
                      Printf.sprintf "csel %s, %s, %s, %s" (reg_ref "d" g)
                        (reg_ref "a" g) (reg_ref "b" g) (Cond.name c);
                  };
                ])
              conds
          @ [
              {
                Form.name = "cbz." ^ n;
                inputs = [ g ];
                result = Some (G Sz.X);
                asm =
                  Printf.sprintf
                    "mov %%x[d], #0\n\
                     \tcbz %s, 1f\n\
                     \tb 2f\n\
                     1:\tmov %%x[d], #1\n\
                     2:"
                    (reg_ref "a" g);
                mem = false;
                make = (fun vs _ -> `Test (Cbz (sz, arg vs 0)));
              };
              {
                Form.name = "cbnz." ^ n;
                inputs = [ g ];
                result = Some (G Sz.X);
                asm =
                  Printf.sprintf
                    "mov %%x[d], #0\n\
                     \tcbnz %s, 1f\n\
                     \tb 2f\n\
                     1:\tmov %%x[d], #1\n\
                     2:"
                    (reg_ref "a" g);
                mem = false;
                make = (fun vs _ -> `Test (Cbnz (sz, arg vs 0)));
              };
            ])
        szs;
      List.map
        (fun c ->
          {
            (form ~inputs:[]
               ("cset." ^ Cond.name c)
               (G Sz.W) "cset"
               (fun _ ->
                 Cset
                   ( c,
                     Mir_value.
                       { id = Mir_id.Value.of_int 9; ty = Mir_type.Flags } )))
            with
            Form.asm = Printf.sprintf "cset %%w[d], %s" (Cond.name c);
          })
        conds;
      List.map
        (fun c ->
          {
            Form.name = "b." ^ Cond.name c;
            inputs = [];
            result = Some (G Sz.X);
            asm =
              Printf.sprintf
                "mov %%x[d], #0\n\tb.%s 1f\n\tb 2f\n1:\tmov %%x[d], #1\n2:"
                (Cond.name c);
            mem = false;
            make = (fun _ f -> `Test (B_cond (c, f)));
          })
        conds;
      List.concat_map
        (fun (t, sz, imm, shift) ->
          let g = G sz in
          [
            {
              (form ~inputs:[]
                 (Printf.sprintf "movz.%s#0x%x<<%d" (sz_name sz) imm shift)
                 g "movz"
                 (fun _ -> Movz (t, imm, shift)))
              with
              Form.asm =
                Printf.sprintf "movz %s, #0x%x, lsl %d" (reg_ref "d" g) imm
                  shift;
            };
            {
              (form ~inputs:[]
                 (Printf.sprintf "movn.%s#0x%x<<%d" (sz_name sz) imm shift)
                 g "movn"
                 (fun _ -> Movn (sz, imm, shift)))
              with
              Form.asm =
                Printf.sprintf "movn %s, #0x%x, lsl %d" (reg_ref "d" g) imm
                  shift;
            };
            {
              (form ~inputs:[ g ]
                 (Printf.sprintf "movk.%s#0x%x<<%d" (sz_name sz) imm shift)
                 g "movk"
                 (fun vs -> Movk (sz, arg vs 0, imm, shift)))
              with
              (* the tie: the destination register is the input register *)
              Form.asm =
                Printf.sprintf "mov %s, %s\n\tmovk %s, #0x%x, lsl %d"
                  (reg_ref "d" g) (reg_ref "a" g) (reg_ref "d" g) imm shift;
            };
          ])
        [
          (Mir_type.i32, Sz.W, 0x1234, 0);
          (Mir_type.i32, Sz.W, 0xFFFF, 16);
          (Mir_type.i64, Sz.X, 0xBEEF, 48);
          (Mir_type.i64, Sz.X, 0x8000, 32);
        ];
      [
        {
          (form ~inputs:[ G Sz.W ] "sxtw" (G Sz.X) "sxtw" (fun vs ->
               Sxtw (arg vs 0)))
          with
          Form.asm = "sxtw %x[d], %w[a]";
        };
        {
          (form ~inputs:[ G Sz.W ] "uxtw" (G Sz.X) "mov" (fun vs ->
               Uxtw (arg vs 0)))
          with
          Form.asm = "mov %w[d], %w[a]";
        };
        {
          (form ~inputs:[ G Sz.X ] "wtrunc" (G Sz.W) "mov" (fun vs ->
               Wtrunc (arg vs 0)))
          with
          Form.asm = "mov %w[d], %w[a]";
        };
      ];
      List.concat_map
        (fun fsz ->
          let f = F fsz and n = fsz_name fsz in
          let other = match fsz with Fsz.D -> F Fsz.S | Fsz.S -> F Fsz.D in
          let gpr = match fsz with Fsz.D -> G Sz.X | Fsz.S -> G Sz.W in
          List.map
            (fun (o, on) ->
              form
                (on ^ "." ^ n)
                f on
                (fun vs -> Fbin (o, fsz, arg vs 0, arg vs 1)))
            [
              (Fop.Add, "fadd");
              (Fop.Div, "fdiv");
              (Fop.Max, "fmax");
              (Fop.Mul, "fmul");
              (Fop.Sub, "fsub");
            ]
          @ List.map
              (fun (u, un) ->
                form ~inputs:[ f ]
                  (un ^ "." ^ n)
                  f un
                  (fun vs -> Funary (u, fsz, arg vs 0)))
              [
                (Funary.Fneg, "fneg");
                (Funary.Frintz, "frintz");
                (Funary.Fsqrt, "fsqrt");
              ]
          @ [
              {
                (form ~result:false ("fcmp." ^ n) f "fcmp" (fun vs ->
                     Fcmp (fsz, arg vs 0, arg vs 1)))
                with
                Form.result = None;
              };
              form ~inputs:[ f; f; f ] ("fmadd." ^ n) f "fmadd" (fun vs ->
                  Fmadd (fsz, arg vs 0, arg vs 1, arg vs 2));
              form ~inputs:[ f ] ("fmov." ^ n) f "fmov" (fun vs ->
                  Fmov (fsz, arg vs 0));
              form ~inputs:[ other ] ("fcvt." ^ n) f "fcvt" (fun vs ->
                  Fcvt (fsz, arg vs 0));
              {
                (form ~inputs:[ f ] ("fcvtzs.x." ^ n) (G Sz.X) "fcvtzs"
                   (fun vs -> Fcvtzs (fsz, arg vs 0)))
                with
                Form.asm = Printf.sprintf "fcvtzs %%x[d], %s" (reg_ref "a" f);
              };
              {
                (form ~inputs:[ G Sz.X ] ("scvtf." ^ n) f "scvtf" (fun vs ->
                     Scvtf (fsz, arg vs 0)))
                with
                Form.asm = Printf.sprintf "scvtf %s, %%x[a]" (reg_ref "d" f);
              };
              form ~inputs:[ gpr ] ("fmov.from_gpr." ^ n) f "fmov" (fun vs ->
                  Fmov_from_gpr (fsz, arg vs 0));
              form ~inputs:[ f ] ("fmov.to_gpr." ^ n) gpr "fmov" (fun vs ->
                  Fmov_to_gpr (fsz, arg vs 0));
            ]
          @ List.map
              (fun c ->
                {
                  (form
                     ("fcsel." ^ n ^ "." ^ Cond.name c)
                     f "fcsel"
                     (fun vs ->
                       Fcsel
                         ( fsz,
                           c,
                           Mir_value.
                             { id = Mir_id.Value.of_int 9; ty = Mir_type.Flags },
                           arg vs 0,
                           arg vs 1 )))
                  with
                  Form.asm =
                    Printf.sprintf "fcsel %s, %s, %s, %s" (reg_ref "d" f)
                      (reg_ref "a" f) (reg_ref "b" f) (Cond.name c);
                })
              Cond.[ Eq; Mi; Ls; Vs; Ne; Gt ])
        fszs;
      List.concat_map
        (fun (m, r) ->
          List.concat_map
            (fun k ->
              let size = Msz.bytes m in
              let off = Int64.mul size k in
              [
                {
                  Form.name = Printf.sprintf "ldr.%s#%Ld" (Msz.name m) off;
                  inputs = [];
                  result = Some r;
                  asm =
                    Printf.sprintf "ldr%s %s, [%%x[base], #%Ld]"
                      (match m with Msz.B -> "b" | Msz.H -> "h" | _ -> "")
                      (reg_ref "d" r) off;
                  mem = true;
                  make =
                    (fun _ _ ->
                      `Op
                        (Ldr
                           ( m,
                             Mir_value.
                               { id = Mir_id.Value.of_int 8; ty = Mir_type.Ptr },
                             off )));
                };
                {
                  Form.name = Printf.sprintf "str.%s#%Ld" (Msz.name m) off;
                  inputs = [ r ];
                  result = None;
                  asm =
                    Printf.sprintf "str%s %s, [%%x[base], #%Ld]"
                      (match m with Msz.B -> "b" | Msz.H -> "h" | _ -> "")
                      (reg_ref "a" r) off;
                  mem = true;
                  make =
                    (fun vs _ ->
                      `Op
                        (Str
                           ( m,
                             Mir_value.
                               { id = Mir_id.Value.of_int 8; ty = Mir_type.Ptr },
                             off,
                             List.hd vs )));
                };
              ])
            [ 0L; 1L ])
        [
          (Msz.B, G Sz.W);
          (Msz.H, G Sz.W);
          (Msz.W, G Sz.W);
          (Msz.X, G Sz.X);
          (Msz.S, F Fsz.S);
          (Msz.D, F Fsz.D);
        ];
      List.map
        (fun (signed, from, name) ->
          {
            (form ~inputs:[ G Sz.W ] name (G Sz.W) name (fun vs ->
                 Ext
                   {
                     signed;
                     from;
                     src = { (arg vs 0) with Mir_value.ty = Mir_type.Int from };
                   }))
            with
            Form.asm = Printf.sprintf "%s %%w[d], %%w[a]" name;
          })
        [
          (true, Mir_width.W8, "sxtb");
          (true, Mir_width.W16, "sxth");
          (false, Mir_width.W8, "uxtb");
          (false, Mir_width.W16, "uxth");
        ];
      List.map
        (fun (w, name) ->
          {
            (form ~inputs:[ G Sz.W ] (name ^ ".trunc") (G Sz.W) name (fun vs ->
                 Trunc (w, arg vs 0)))
            with
            Form.asm = Printf.sprintf "%s %%w[d], %%w[a]" name;
          })
        [ (Mir_width.W8, "uxtb"); (Mir_width.W16, "uxth") ];
    ]
