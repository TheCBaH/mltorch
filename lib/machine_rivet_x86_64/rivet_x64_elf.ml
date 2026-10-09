(* A bound image as a static x86-64 ELF executable: one PT_LOAD per segment,
   file offsets congruent to their addresses modulo the page, the entry at the
   image's entry. Nothing is linked: Rivet has already placed every byte. *)

let le n v =
  String.init n (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xffL)))

let u16 v = le 2 (Int64.of_int v)
let u32 v = le 4 (Int64.of_int v)
let u64 v = le 8 v
let ehsize = 64
let phsize = 56
let page = 4096

let flags (p : Asm_core.Perms.t) =
  (if p.Asm_core.Perms.execute then 1 else 0)
  lor (if p.Asm_core.Perms.write then 2 else 0)
  lor if p.Asm_core.Perms.read then 4 else 0

let write (image : Image.t) =
  let segments =
    List.filter
      (fun (s : Image.segment) ->
        String.length s.Image.bytes > 0 || s.Image.zero_fill > 0)
      image.Image.segments
  in
  let n = List.length segments in
  let first = ehsize + (n * phsize) in
  (* each segment's bytes start at a file offset whose residue modulo the page
     is the address's *)
  let placed, _ =
    List.fold_left
      (fun (acc, at) (s : Image.segment) ->
        let want =
          Int64.to_int (Int64.rem s.Image.address (Int64.of_int page))
        in
        let off = ((at + page - 1) / page * page) + want in
        let off = if off < at then off + page else off in
        ((s, off) :: acc, off + String.length s.Image.bytes))
      ([], first) segments
  in
  let placed = List.rev placed in
  let entry = Option.value image.Image.entry ~default:0L in
  let header =
    "\x7fELF" ^ "\x02\x01\x01\x00" ^ String.make 8 '\000' ^ u16 2 ^ u16 62
    ^ u32 1 ^ u64 entry
    ^ u64 (Int64.of_int ehsize)
    ^ u64 0L ^ u32 0 ^ u16 ehsize ^ u16 phsize ^ u16 n ^ u16 64 ^ u16 0 ^ u16 0
  in
  let phdrs =
    String.concat ""
      (List.map
         (fun ((s : Image.segment), off) ->
           u32 1
           ^ u32 (flags s.Image.perms)
           ^ u64 (Int64.of_int off)
           ^ u64 s.Image.address ^ u64 s.Image.address
           ^ u64 (Int64.of_int (String.length s.Image.bytes))
           ^ u64
               (Int64.of_int (String.length s.Image.bytes + s.Image.zero_fill))
           ^ u64 (Int64.of_int page))
         placed)
  in
  let buf = Buffer.create 65536 in
  Buffer.add_string buf header;
  Buffer.add_string buf phdrs;
  List.iter
    (fun ((s : Image.segment), off) ->
      Buffer.add_string buf (String.make (off - Buffer.length buf) '\000');
      Buffer.add_string buf s.Image.bytes)
    placed;
  Buffer.contents buf
