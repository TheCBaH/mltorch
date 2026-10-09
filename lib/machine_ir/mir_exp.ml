(* The project-owned binary64 [exp]: the algorithm and the constants of
   glibc 2.41's table-driven [exp] (N = 128, a degree-5 polynomial), so that
   it agrees bit for bit with the host libm the reference calls, and a native
   realization can carry it without linking that library. The operation order
   is the one the AArch64 library executes, with a fused multiply-add where
   it fuses; another libm that rounds differently is a documented
   disagreement, never a tolerance. [model] is the specification the native
   realizations are checked against. *)

let bits = Int64.bits_of_float
let of_bits = Int64.float_of_bits

(* [N / ln 2], [-ln 2 / N] as a high and a low part, then the coefficients of
   degree 2 to 5. *)
let invln2n = of_bits 0x40671547652b82feL
let neg_ln2hi_n = of_bits 0xbf762e42fefa0000L
let neg_ln2lo_n = of_bits 0xbd0cf79abc9e3b3aL
let c2 = of_bits 0x3fdffffffffffdbdL
let c3 = of_bits 0x3fc555555555543cL
let c4 = of_bits 0x3fa55555cf172b91L
let c5 = of_bits 0x3f81111167a4d017L

(* The seven constants in the order above, as a native realization lays them
   out ahead of {!table}. *)
let constants =
  Array.map bits [| invln2n; neg_ln2hi_n; neg_ln2lo_n; c2; c3; c4; c5 |]

(* Pairs [tail; scale bits] for [2^(i/128)], [i] in [0, 128): the scale bits
   have [i lsl 45] subtracted, so adding [k lsl 45] puts [k] in the exponent. *)
let table =
  [|
    0x0000000000000000L;
    0x3ff0000000000000L;
    0x3c9b3b4f1a88bf6eL;
    0x3feff63da9fb3335L;
    0xbc7160139cd8dc5dL;
    0x3fefec9a3e778061L;
    0xbc905e7a108766d1L;
    0x3fefe315e86e7f85L;
    0x3c8cd2523567f613L;
    0x3fefd9b0d3158574L;
    0xbc8bce8023f98efaL;
    0x3fefd06b29ddf6deL;
    0x3c60f74e61e6c861L;
    0x3fefc74518759bc8L;
    0x3c90a3e45b33d399L;
    0x3fefbe3ecac6f383L;
    0x3c979aa65d837b6dL;
    0x3fefb5586cf9890fL;
    0x3c8eb51a92fdeffcL;
    0x3fefac922b7247f7L;
    0x3c3ebe3d702f9cd1L;
    0x3fefa3ec32d3d1a2L;
    0xbc6a033489906e0bL;
    0x3fef9b66affed31bL;
    0xbc9556522a2fbd0eL;
    0x3fef9301d0125b51L;
    0xbc5080ef8c4eea55L;
    0x3fef8abdc06c31ccL;
    0xbc91c923b9d5f416L;
    0x3fef829aaea92de0L;
    0x3c80d3e3e95c55afL;
    0x3fef7a98c8a58e51L;
    0xbc801b15eaa59348L;
    0x3fef72b83c7d517bL;
    0xbc8f1ff055de323dL;
    0x3fef6af9388c8deaL;
    0x3c8b898c3f1353bfL;
    0x3fef635beb6fcb75L;
    0xbc96d99c7611eb26L;
    0x3fef5be084045cd4L;
    0x3c9aecf73e3a2f60L;
    0x3fef54873168b9aaL;
    0xbc8fe782cb86389dL;
    0x3fef4d5022fcd91dL;
    0x3c8a6f4144a6c38dL;
    0x3fef463b88628cd6L;
    0x3c807a05b0e4047dL;
    0x3fef3f49917ddc96L;
    0x3c968efde3a8a894L;
    0x3fef387a6e756238L;
    0x3c875e18f274487dL;
    0x3fef31ce4fb2a63fL;
    0x3c80472b981fe7f2L;
    0x3fef2b4565e27cddL;
    0xbc96b87b3f71085eL;
    0x3fef24dfe1f56381L;
    0x3c82f7e16d09ab31L;
    0x3fef1e9df51fdee1L;
    0xbc3d219b1a6fbffaL;
    0x3fef187fd0dad990L;
    0x3c8b3782720c0ab4L;
    0x3fef1285a6e4030bL;
    0x3c6e149289cecb8fL;
    0x3fef0cafa93e2f56L;
    0x3c834d754db0abb6L;
    0x3fef06fe0a31b715L;
    0x3c864201e2ac744cL;
    0x3fef0170fc4cd831L;
    0x3c8fdd395dd3f84aL;
    0x3feefc08b26416ffL;
    0xbc86a3803b8e5b04L;
    0x3feef6c55f929ff1L;
    0xbc924aedcc4b5068L;
    0x3feef1a7373aa9cbL;
    0xbc9907f81b512d8eL;
    0x3feeecae6d05d866L;
    0xbc71d1e83e9436d2L;
    0x3feee7db34e59ff7L;
    0xbc991919b3ce1b15L;
    0x3feee32dc313a8e5L;
    0x3c859f48a72a4c6dL;
    0x3feedea64c123422L;
    0xbc9312607a28698aL;
    0x3feeda4504ac801cL;
    0xbc58a78f4817895bL;
    0x3feed60a21f72e2aL;
    0xbc7c2c9b67499a1bL;
    0x3feed1f5d950a897L;
    0x3c4363ed60c2ac11L;
    0x3feece086061892dL;
    0x3c9666093b0664efL;
    0x3feeca41ed1d0057L;
    0x3c6ecce1daa10379L;
    0x3feec6a2b5c13cd0L;
    0x3c93ff8e3f0f1230L;
    0x3feec32af0d7d3deL;
    0x3c7690cebb7aafb0L;
    0x3feebfdad5362a27L;
    0x3c931dbdeb54e077L;
    0x3feebcb299fddd0dL;
    0xbc8f94340071a38eL;
    0x3feeb9b2769d2ca7L;
    0xbc87deccdc93a349L;
    0x3feeb6daa2cf6642L;
    0xbc78dec6bd0f385fL;
    0x3feeb42b569d4f82L;
    0xbc861246ec7b5cf6L;
    0x3feeb1a4ca5d920fL;
    0x3c93350518fdd78eL;
    0x3feeaf4736b527daL;
    0x3c7b98b72f8a9b05L;
    0x3feead12d497c7fdL;
    0x3c9063e1e21c5409L;
    0x3feeab07dd485429L;
    0x3c34c7855019c6eaL;
    0x3feea9268a5946b7L;
    0x3c9432e62b64c035L;
    0x3feea76f15ad2148L;
    0xbc8ce44a6199769fL;
    0x3feea5e1b976dc09L;
    0xbc8c33c53bef4da8L;
    0x3feea47eb03a5585L;
    0xbc845378892be9aeL;
    0x3feea34634ccc320L;
    0xbc93cedd78565858L;
    0x3feea23882552225L;
    0x3c5710aa807e1964L;
    0x3feea155d44ca973L;
    0xbc93b3efbf5e2228L;
    0x3feea09e667f3bcdL;
    0xbc6a12ad8734b982L;
    0x3feea012750bdabfL;
    0xbc6367efb86da9eeL;
    0x3fee9fb23c651a2fL;
    0xbc80dc3d54e08851L;
    0x3fee9f7df9519484L;
    0xbc781f647e5a3ecfL;
    0x3fee9f75e8ec5f74L;
    0xbc86ee4ac08b7db0L;
    0x3fee9f9a48a58174L;
    0xbc8619321e55e68aL;
    0x3fee9feb564267c9L;
    0x3c909ccb5e09d4d3L;
    0x3feea0694fde5d3fL;
    0xbc7b32dcb94da51dL;
    0x3feea11473eb0187L;
    0x3c94ecfd5467c06bL;
    0x3feea1ed0130c132L;
    0x3c65ebe1abd66c55L;
    0x3feea2f336cf4e62L;
    0xbc88a1c52fb3cf42L;
    0x3feea427543e1a12L;
    0xbc9369b6f13b3734L;
    0x3feea589994cce13L;
    0xbc805e843a19ff1eL;
    0x3feea71a4623c7adL;
    0xbc94d450d872576eL;
    0x3feea8d99b4492edL;
    0x3c90ad675b0e8a00L;
    0x3feeaac7d98a6699L;
    0x3c8db72fc1f0eab4L;
    0x3feeace5422aa0dbL;
    0xbc65b6609cc5e7ffL;
    0x3feeaf3216b5448cL;
    0x3c7bf68359f35f44L;
    0x3feeb1ae99157736L;
    0xbc93091fa71e3d83L;
    0x3feeb45b0b91ffc6L;
    0xbc5da9b88b6c1e29L;
    0x3feeb737b0cdc5e5L;
    0xbc6c23f97c90b959L;
    0x3feeba44cbc8520fL;
    0xbc92434322f4f9aaL;
    0x3feebd829fde4e50L;
    0xbc85ca6cd7668e4bL;
    0x3feec0f170ca07baL;
    0x3c71affc2b91ce27L;
    0x3feec49182a3f090L;
    0x3c6dd235e10a73bbL;
    0x3feec86319e32323L;
    0xbc87c50422622263L;
    0x3feecc667b5de565L;
    0x3c8b1c86e3e231d5L;
    0x3feed09bec4a2d33L;
    0xbc91bbd1d3bcbb15L;
    0x3feed503b23e255dL;
    0x3c90cc319cee31d2L;
    0x3feed99e1330b358L;
    0x3c8469846e735ab3L;
    0x3feede6b5579fdbfL;
    0xbc82dfcd978e9db4L;
    0x3feee36bbfd3f37aL;
    0x3c8c1a7792cb3387L;
    0x3feee89f995ad3adL;
    0xbc907b8f4ad1d9faL;
    0x3feeee07298db666L;
    0xbc55c3d956dcaebaL;
    0x3feef3a2b84f15fbL;
    0xbc90a40e3da6f640L;
    0x3feef9728de5593aL;
    0xbc68d6f438ad9334L;
    0x3feeff76f2fb5e47L;
    0xbc91eee26b588a35L;
    0x3fef05b030a1064aL;
    0x3c74ffd70a5fddcdL;
    0x3fef0c1e904bc1d2L;
    0xbc91bdfbfa9298acL;
    0x3fef12c25bd71e09L;
    0x3c736eae30af0cb3L;
    0x3fef199bdd85529cL;
    0x3c8ee3325c9ffd94L;
    0x3fef20ab5fffd07aL;
    0x3c84e08fd10959acL;
    0x3fef27f12e57d14bL;
    0x3c63cdaf384e1a67L;
    0x3fef2f6d9406e7b5L;
    0x3c676b2c6c921968L;
    0x3fef3720dcef9069L;
    0xbc808a1883ccb5d2L;
    0x3fef3f0b555dc3faL;
    0xbc8fad5d3ffffa6fL;
    0x3fef472d4a07897cL;
    0xbc900dae3875a949L;
    0x3fef4f87080d89f2L;
    0x3c74a385a63d07a7L;
    0x3fef5818dcfba487L;
    0xbc82919e2040220fL;
    0x3fef60e316c98398L;
    0x3c8e5a50d5c192acL;
    0x3fef69e603db3285L;
    0x3c843a59ac016b4bL;
    0x3fef7321f301b460L;
    0xbc82d52107b43e1fL;
    0x3fef7c97337b9b5fL;
    0xbc892ab93b470dc9L;
    0x3fef864614f5a129L;
    0x3c74b604603a88d3L;
    0x3fef902ee78b3ff6L;
    0x3c83c5ec519d7271L;
    0x3fef9a51fbc74c83L;
    0xbc8ff7128fd391f0L;
    0x3fefa4afa2a490daL;
    0xbc8dae98e223747dL;
    0x3fefaf482d8e67f1L;
    0x3c8ec3bc41aa2008L;
    0x3fefba1bee615a27L;
    0x3c842b94c3a9eb32L;
    0x3fefc52b376bba97L;
    0x3c8a64a931d185eeL;
    0x3fefd0765b6e4540L;
    0xbc8e37bae43be3edL;
    0x3fefdbfdad9cbe14L;
    0x3c77893b4d91cd9dL;
    0x3fefe7c1819e90d8L;
    0x3c5305c14160cc89L;
    0x3feff3c22b8f71f1L;
  |]

let top12 x = Int64.to_int (Int64.shift_right_logical (bits x) 52) land 0x7ff

(* For [|x| >= 512]: the scale's exponent is adjusted so it cannot overflow
   or underflow, and the result is scaled back, rounding once into the
   subnormal range. *)
let special_case tmp sbits ki =
  if Int64.logand ki 0x80000000L = 0L then
    let scale = of_bits (Int64.sub sbits (Int64.shift_left 1009L 52)) in
    0x1p1009 *. Float.fma tmp scale scale
  else
    let scale = of_bits (Int64.add sbits (Int64.shift_left 1022L 52)) in
    let st = tmp *. scale in
    let y = scale +. st in
    if y < 1.0 then
      let lo = scale -. y +. st in
      let hi = 1.0 +. y in
      let lo = 1.0 -. hi +. y +. lo in
      let y = hi +. lo -. 1.0 in
      if y = 0.0 then 0.0 else 0x1p-1022 *. y
    else 0x1p-1022 *. y

let model x =
  let top = top12 x in
  if top < 0x3c9 then 1.0 +. x
  else if top >= 0x409 then
    if bits x = 0xfff0000000000000L then 0.0
    else if top = 0x7ff then 1.0 +. x
    else if Int64.compare (bits x) 0L < 0 then 0.0
    else Float.infinity
  else
    let z = x *. invln2n in
    let kd = Float.round z in
    let ki = Int64.of_float kd in
    let r = Float.fma neg_ln2hi_n kd x in
    let r = Float.fma neg_ln2lo_n kd r in
    let idx = 2 * Int64.to_int (Int64.logand ki 127L) in
    let tail = of_bits table.(idx) in
    let sbits = Int64.add table.(idx + 1) (Int64.shift_left ki 45) in
    let p2 = Float.fma c3 r c2 in
    let r2 = r *. r in
    let p4 = Float.fma c5 r c4 in
    let t = r +. tail in
    let t = Float.fma p2 r2 t in
    let t = Float.fma (r2 *. r2) p4 t in
    if top = 0x408 then special_case t sbits ki
    else
      let scale = of_bits sbits in
      Float.fma t scale scale
