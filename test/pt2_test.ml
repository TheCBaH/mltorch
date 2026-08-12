(* Hermetic unit tests for the pure-OCaml pt2 reader. No network, no libtorch:
   we build a tiny STORED zip and a minimal tensor pickle in memory and check
   the reader round-trips them. *)

let le16 b v =
  Buffer.add_char b (Char.chr (v land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 8) land 0xff))

let le32 b v =
  le16 b (v land 0xffff);
  le16 b ((v lsr 16) land 0xffff)

(* Build a ZIP archive from [(name, data)] entries (STORED) via zipc — the same
   writer our reader is built on, so it exercises a real, CRC-correct archive. *)
let make_zip entries =
  let add z (path, data) =
    let file =
      match Zipc.File.stored_of_binary_string data with
      | Ok file -> file
      | Error e -> failwith ("Zipc.File.stored_of_binary_string: " ^ e)
    in
    let m =
      match Zipc.Member.make ~path (Zipc.Member.File file) with
      | Ok m -> m
      | Error e -> failwith ("Zipc.Member.make: " ^ e)
    in
    Zipc.add m z
  in
  match Zipc.to_binary_string (List.fold_left add Zipc.empty entries) with
  | Ok s -> s
  | Error e -> failwith ("Zipc.to_binary_string: " ^ e)

let%expect_test "zip round-trips STORED entries (incl. relative prefix)" =
  match
    Pt2_zip.of_string
      (make_zip
         [ ("m/models/model.json", "{\"k\":1}"); ("m/data/0", "raw-bytes") ])
  with
  | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e)
  | Ok zip ->
      Printf.printf "prefix=%s\n" (Pt2_zip.prefix zip);
      (match Pt2_zip.read_rel_required zip "models/model.json" with
      | Ok s -> Printf.printf "model.json=%s\n" s
      | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e));
      (match Pt2_zip.read_rel_required zip "data/0" with
      | Ok s -> Printf.printf "data/0=%s\n" s
      | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e));
      (match Pt2_zip.read zip "m/nope" with
      | Ok missing -> Printf.printf "missing=%b\n" (missing = None)
      | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e));
      [%expect
        {|
    prefix=m
    model.json={"k":1}
    data/0=raw-bytes
    missing=true
    |}]

let%expect_test "zip missing entry is typed" =
  match Pt2_zip.of_string (make_zip [ ("m/data/0", "raw-bytes") ]) with
  | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e)
  | Ok zip ->
      (match Pt2_zip.read_rel_required zip "models/model.json" with
      | Ok _ -> print_endline "unexpected success"
      | Error e -> Format.printf "%a@." Pt2_zip.pp_error (Err.Error.kind e));
      [%expect {| zip entry "m/models/model.json" is missing |}]

(* --- Pt2_tensor --- *)

let%expect_test "metadata: numel, contiguity, pp" =
  let mk strides =
    {
      Pt2_tensor.dtype = Pt2_dtype.Float32;
      sizes = [ 2; 3 ];
      strides;
      storage_offset = 0;
      data = Bytes.create (6 * 4);
    }
  in
  let c = mk [ 3; 1 ]
  (* row-major *)
  and s =
    mk [ 1; 2 ]
    (* column-major *)
  in
  Format.printf "%a numel=%d contig(c)=%b contig(s)=%b@." Pt2_tensor.pp c
    (Pt2_tensor.numel c)
    (Pt2_tensor.is_contiguous c)
    (Pt2_tensor.is_contiguous s);
  [%expect {| float32[2; 3] numel=6 contig(c)=true contig(s)=false |}]

let%expect_test "dtype and tensor metadata failures are typed" =
  (match Pt2_dtype.of_scalar_type Pytorch_types.ScalarType.HALF with
  | Ok _ -> print_endline "unexpected success"
  | Error e -> Format.printf "%a@." Pt2_dtype.pp_error (Err.Error.kind e));
  Pt2_tensor.int_of_symint ~field:"sizes"
    (Pytorch_types.SymInt.Expr (Pytorch_types.SymExpr.make "s0" None))
  |> Format.printf "%a@."
       (Core.Pretty.err_result
          ~ok:(Fmt.any "unexpected success")
          ~error:Pt2_tensor.pp_error);
  [%expect
    {|
    unsupported ScalarType HALF
    symbolic tensor metadata is unsupported for sizes
    |}]

(* --- Pt2_pickle --- *)

(* Hand-encode the pickle torch.save emits for one tensor:
   _rebuild_tensor_v2((storage, FloatStorage, '0', 'cpu', 4), 0, (2,2), (2,1),
   False, OrderedDict()). *)
let tensor_pickle () =
  let b = Buffer.create 128 in
  let op c = Buffer.add_char b (Char.chr c) in
  let global m n =
    op 0x63;
    Buffer.add_string b (m ^ "\n" ^ n ^ "\n")
  in
  let unicode s =
    op 0x58;
    le32 b (String.length s);
    Buffer.add_string b s
  in
  let int1 v =
    op 0x4b;
    Buffer.add_char b (Char.chr v)
  in
  op 0x80;
  Buffer.add_char b '\002' (* PROTO 2 *);
  global "torch._utils" "_rebuild_tensor_v2";
  op 0x28 (* MARK: start args *);
  op 0x28 (* MARK: start persid tuple *);
  unicode "storage";
  global "torch" "FloatStorage";
  unicode "0";
  unicode "cpu";
  int1 4;
  op 0x74 (* TUPLE -> persid *);
  op 0x51 (* BINPERSID *);
  int1 0 (* storage_offset *);
  op 0x28;
  int1 2;
  int1 2;
  op 0x74 (* size tuple (2,2) *);
  op 0x28;
  int1 2;
  int1 1;
  op 0x74 (* stride tuple (2,1) *);
  op 0x89 (* NEWFALSE: requires_grad *);
  op 0x7d (* EMPTY_DICT: hooks *);
  op 0x74 (* TUPLE -> args *);
  op 0x52 (* REDUCE *);
  op 0x2e (* STOP *);
  Buffer.contents b

let%expect_test "pickle yields rebuild descriptor" =
  match Pt2_pickle.parse_tensor (tensor_pickle ()) with
  | Error e -> Format.printf "%a@." Pt2_pickle.pp_error (Err.Error.kind e)
  | Ok rb ->
      Printf.printf "key=%s dtype=%s offset=%d sizes=[%s] strides=[%s]\n"
        rb.Pt2_pickle.storage_key
        (Pt2_dtype.to_string rb.Pt2_pickle.dtype)
        rb.Pt2_pickle.storage_offset
        (String.concat ";" (List.map string_of_int rb.Pt2_pickle.sizes))
        (String.concat ";" (List.map string_of_int rb.Pt2_pickle.strides));
      [%expect {| key=0 dtype=float32 offset=0 sizes=[2;2] strides=[2;1] |}]

let%expect_test "pickle failures are typed" =
  match Pt2_pickle.parse_tensor "not a pickle" with
  | Ok _ -> print_endline "unexpected success"
  | Error e ->
      Format.printf "%a@." Pt2_pickle.pp_error (Err.Error.kind e);
      [%expect
        {| pickle decode failed: pickle error at byte 1: unknown opcode 0x6e |}]

(* ---------------------------------------------------------------------------
   32-bit int portability. These two blocks are the reason this suite carries
   `(modes best js)`: each covers a narrowing whose CORRECT behaviour differs
   between a 63-bit native int and js_of_ocaml's 32-bit one, so neither can be
   pinned by a golden that spells out one backend's answer.

   The shape used throughout: compute what the answer OUGHT to be on whichever
   backend is running, compare the real answer to it, and print only whether
   they agree. The golden reads `true` everywhere, and a regression on either
   backend still flips it. See .ai/js_backends_design.md.
   ------------------------------------------------------------------------- *)

(* [Opickle.Src.uint4] reads an unsigned 32-bit word. [Int32.unsigned_to_int]
   encodes exactly the right notion of "what should this word mean here",
   including returning None where no int can hold the answer, so it is the
   oracle.

   The range above 2^31 is the interesting half, and it only became assertable
   once opickle was fixed. It used to build the value with `land 0xFFFF_FFFF`,
   a literal that IS -1 wherever int is 32 bits -- js_of_ocaml says so itself,
   "integer 0xffffffff truncated to -1" -- so the mask became a no-op and a word
   with the high bit set came back as a negative length. Natively the same word
   was correctly 4294967295, and those two answers cannot both be a green
   [%expect]: one golden, two backends, and there the difference WAS the bug.

   Now that uint4 raises where the value is unrepresentable, the two backends
   agree again -- by both matching the oracle rather than by returning the same
   number -- so the boundary words belong here. They are the cases that would
   regress first if the mask ever came back. See .ai/js_backends_design.md. *)
let uint4_agrees word_le =
  let expected = Int32.unsigned_to_int (String.get_int32_le word_le 0) in
  match
    Opickle.Src.uint4
      (Opickle.Src.of_reader (Bytesrw.Bytes.Reader.of_string word_le))
  with
  | got -> ( match expected with Some e -> got = e | None -> false)
  | exception _ -> expected = None

let%expect_test
    "Src.uint4 agrees with Int32.unsigned_to_int over the full range" =
  Printf.printf
    "zero=%b small=%b mid=%b max-signed=%b high-bit=%b all-ones=%b\n"
    (uint4_agrees "\x00\x00\x00\x00")
    (uint4_agrees "\x2a\x00\x00\x00")
    (uint4_agrees "\x00\x00\x00\x40")
    (uint4_agrees "\xff\xff\xff\x7f")
    (uint4_agrees "\x00\x00\x00\x80")
    (uint4_agrees "\xff\xff\xff\xff");
  [%expect
    {| zero=true small=true mid=true max-signed=true high-bit=true all-ones=true |}]

(* [Pt2_pickle.int_of] narrows a signed pickle integer. Pickle ints are signed,
   so the bound is two-sided -- a value below min_int wraps exactly as one above
   max_int does, and only checking the top would leave half the defect in place. *)
let int_of_agrees i64 =
  let representable =
    Int64.compare i64 (Int64.of_int min_int) >= 0
    && Int64.compare i64 (Int64.of_int max_int) <= 0
  in
  match Pt2_pickle.int_of "size" (Opickle.Value.Int i64) with
  | Ok v -> representable && Int64.equal (Int64.of_int v) i64
  | Error _ -> not representable

let%expect_test "Pt2_pickle.int_of bounds both ends" =
  Printf.printf "zero=%b small=%b negative=%b max=%b min=%b\n"
    (int_of_agrees 0L) (int_of_agrees 42L) (int_of_agrees (-42L))
    (int_of_agrees Int64.max_int)
    (int_of_agrees Int64.min_int);
  (* 2^40 is representable natively and NOT under js_of_ocaml -- the one case
     that actually differs between the two, and the reason for the oracle. *)
  Printf.printf "2^40=%b -2^40=%b\n"
    (int_of_agrees 1099511627776L)
    (int_of_agrees (-1099511627776L));
  [%expect
    {|
    zero=true small=true negative=true max=true min=true
    2^40=true -2^40=true |}]

(* An aggregate bound has to check the MULTIPLY, not just its result. Folding
   [max_int; 4] in int64 wraps to -4, which passes any range test on the final
   value and yields a nonsense numel -- so this asserts the raise, which is the
   behaviour on both backends once the step itself is checked. The values are
   written in terms of [max_int] rather than a literal so the case stays
   meaningful where int is 32 bits and where it is 63. *)
let raises_invalid_arg f =
  match f () with exception Invalid_argument _ -> true | _ -> false

let%expect_test "Pt2_tensor.numel rejects a product that overflows mid-fold" =
  let mk sizes =
    {
      Pt2_tensor.dtype = Pt2_dtype.Float32;
      sizes;
      strides = List.map (fun _ -> 1) sizes;
      storage_offset = 0;
      data = Bytes.create 0;
    }
  in
  Printf.printf "wraps=%b overflows-int=%b ok=%b\n"
    (raises_invalid_arg (fun () -> Pt2_tensor.numel (mk [ max_int; 4 ])))
    (raises_invalid_arg (fun () -> Pt2_tensor.numel (mk [ max_int; 2 ])))
    (Pt2_tensor.numel (mk [ 2; 3; 4 ]) = 24);
  [%expect {| wraps=true overflows-int=true ok=true |}]
