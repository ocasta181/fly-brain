"""Verbatim raw-bit payloads shared by pinned scalar math providers.

Source profiles and provenance remain documented in operator32.py,
operator64.py, hyperbolic32.py, hyperbolic_math64.py, transcendentals.py,
expression_rng_math.py and stable_math64.py. Keep numerical algorithms and
their exceptional policies in those original owners.
"""

_EXP32_VALUES = r'''
    0x3ff0000000000000ul,0x3feff63da9fb3335ul,0x3fefec9a3e778061ul,0x3fefe315e86e7f85ul,
    0x3fefd9b0d3158574ul,0x3fefd06b29ddf6deul,0x3fefc74518759bc8ul,0x3fefbe3ecac6f383ul,
    0x3fefb5586cf9890ful,0x3fefac922b7247f7ul,0x3fefa3ec32d3d1a2ul,0x3fef9b66affed31bul,
    0x3fef9301d0125b51ul,0x3fef8abdc06c31ccul,0x3fef829aaea92de0ul,0x3fef7a98c8a58e51ul,
    0x3fef72b83c7d517bul,0x3fef6af9388c8deaul,0x3fef635beb6fcb75ul,0x3fef5be084045cd4ul,
    0x3fef54873168b9aaul,0x3fef4d5022fcd91dul,0x3fef463b88628cd6ul,0x3fef3f49917ddc96ul,
    0x3fef387a6e756238ul,0x3fef31ce4fb2a63ful,0x3fef2b4565e27cddul,0x3fef24dfe1f56381ul,
    0x3fef1e9df51fdee1ul,0x3fef187fd0dad990ul,0x3fef1285a6e4030bul,0x3fef0cafa93e2f56ul,
    0x3fef06fe0a31b715ul,0x3fef0170fc4cd831ul,0x3feefc08b26416fful,0x3feef6c55f929ff1ul,
    0x3feef1a7373aa9cbul,0x3feeecae6d05d866ul,0x3feee7db34e59ff7ul,0x3feee32dc313a8e5ul,
    0x3feedea64c123422ul,0x3feeda4504ac801cul,0x3feed60a21f72e2aul,0x3feed1f5d950a897ul,
    0x3feece086061892dul,0x3feeca41ed1d0057ul,0x3feec6a2b5c13cd0ul,0x3feec32af0d7d3deul,
    0x3feebfdad5362a27ul,0x3feebcb299fddd0dul,0x3feeb9b2769d2ca7ul,0x3feeb6daa2cf6642ul,
    0x3feeb42b569d4f82ul,0x3feeb1a4ca5d920ful,0x3feeaf4736b527daul,0x3feead12d497c7fdul,
    0x3feeab07dd485429ul,0x3feea9268a5946b7ul,0x3feea76f15ad2148ul,0x3feea5e1b976dc09ul,
    0x3feea47eb03a5585ul,0x3feea34634ccc320ul,0x3feea23882552225ul,0x3feea155d44ca973ul,
    0x3feea09e667f3bcdul,0x3feea012750bdabful,0x3fee9fb23c651a2ful,0x3fee9f7df9519484ul,
    0x3fee9f75e8ec5f74ul,0x3fee9f9a48a58174ul,0x3fee9feb564267c9ul,0x3feea0694fde5d3ful,
    0x3feea11473eb0187ul,0x3feea1ed0130c132ul,0x3feea2f336cf4e62ul,0x3feea427543e1a12ul,
    0x3feea589994cce13ul,0x3feea71a4623c7adul,0x3feea8d99b4492edul,0x3feeaac7d98a6699ul,
    0x3feeace5422aa0dbul,0x3feeaf3216b5448cul,0x3feeb1ae99157736ul,0x3feeb45b0b91ffc6ul,
    0x3feeb737b0cdc5e5ul,0x3feeba44cbc8520ful,0x3feebd829fde4e50ul,0x3feec0f170ca07baul,
    0x3feec49182a3f090ul,0x3feec86319e32323ul,0x3feecc667b5de565ul,0x3feed09bec4a2d33ul,
    0x3feed503b23e255dul,0x3feed99e1330b358ul,0x3feede6b5579fdbful,0x3feee36bbfd3f37aul,
    0x3feee89f995ad3adul,0x3feeee07298db666ul,0x3feef3a2b84f15fbul,0x3feef9728de5593aul,
    0x3feeff76f2fb5e47ul,0x3fef05b030a1064aul,0x3fef0c1e904bc1d2ul,0x3fef12c25bd71e09ul,
    0x3fef199bdd85529cul,0x3fef20ab5fffd07aul,0x3fef27f12e57d14bul,0x3fef2f6d9406e7b5ul,
    0x3fef3720dcef9069ul,0x3fef3f0b555dc3faul,0x3fef472d4a07897cul,0x3fef4f87080d89f2ul,
    0x3fef5818dcfba487ul,0x3fef60e316c98398ul,0x3fef69e603db3285ul,0x3fef7321f301b460ul,
    0x3fef7c97337b9b5ful,0x3fef864614f5a129ul,0x3fef902ee78b3ff6ul,0x3fef9a51fbc74c83ul,
    0x3fefa4afa2a490daul,0x3fefaf482d8e67f1ul,0x3fefba1bee615a27ul,0x3fefc52b376bba97ul,
    0x3fefd0765b6e4540ul,0x3fefdbfdad9cbe14ul,0x3fefe7c1819e90d8ul,0x3feff3c22b8f71f1ul,
'''

_EXP64_COEFFICIENTS = r'''
    0x401ef1584af17ecdul,0xbfe13b4d4879e55eul,0x4034ffde7cc02cf4ul,0x4038b31b87eb8904ul,
    0x3f55e52272e0eaecul,
'''

_EXP64_VALUES = r'''
    0x3ff0000000000000ul,0x0000000000000000ul,0x3ff0163da9fb3335ul,0x3ca3a4bc88ab6f0bul,
    0x3ff02c9a3e778060ul,0x3cb5448c4f3a3faaul,0x3ff04315e86e7f84ul,0x3cb0cea8d139df85ul,
    0x3ff059b0d3158574ul,0x3c94ca556a1b2fa8ul,0x3ff0706b29ddf6ddul,0x3cb173b2871f2217ul,
    0x3ff0874518759bc8ul,0x3c687a10f84f53b7ul,0x3ff09e3ecac6f383ul,0x3c9801b99f4f5b18ul,
    0x3ff0b5586cf9890ful,0x3ca106d9ba253401ul,0x3ff0cc922b7247f7ul,0x3c96269675ff4e35ul,
    0x3ff0e3ec32d3d1a2ul,0x3c462d2d983e6844ul,0x3ff0fb66affed31aul,0x3cb4935522e05558ul,
    0x3ff11301d0125b50ul,0x3cabdebf5fb67959ul,0x3ff12abdc06c31cbul,0x3cb5245852cc8960ul,
    0x3ff1429aaea92ddful,0x3cadf748a567f8daul,0x3ff15a98c8a58e51ul,0x3c8846f8d141b1d0ul,
    0x3ff172b83c7d517aul,0x3cb24349c9dfc205ul,0x3ff18af9388c8de9ul,0x3caee14f25e66832ul,
    0x3ff1a35beb6fcb75ul,0x3c93dd2c819e0908ul,0x3ff1bbe084045cd3ul,0x3ca92b6b9f4c306aul,
    0x3ff1d4873168b9aaul,0x3ca36c3949dfc34eul,0x3ff1ed5022fcd91cul,0x3cadb25aa6565c95ul,
    0x3ff2063b88628cd6ul,0x3c93118ae698d206ul,0x3ff21f49917ddc96ul,0x3c87c551db68456ful,
    0x3ff2387a6e756238ul,0x3ca045c6a6d3eecbul,0x3ff251ce4fb2a63ful,0x3c90db2bd14acf32ul,
    0x3ff26b4565e27cddul,0x3c877bf4b6a48214ul,0x3ff284dfe1f56380ul,0x3ca77f3dc394bd27ul,
    0x3ff29e9df51fdee1ul,0x3c8b5d8e0885ab1bul,0x3ff2b87fd0dad98ful,0x3cb390730108bc38ul,
    0x3ff2d285a6e4030bul,0x3c93a1fee8c9225cul,0x3ff2ecafa93e2f56ul,0x3c75b2c9eff60911ul,
    0x3ff306fe0a31b715ul,0x3c8bd9046b69ea24ul,0x3ff32170fc4cd831ul,0x3c900e3e54fb7d75ul,
    0x3ff33c08b26416fful,0x3c96fc3180aac683ul,0x3ff356c55f929ff0ul,0x3cae07c0017dfc4aul,
    0x3ff371a7373aa9caul,0x3ca8cb357ddac301ul,0x3ff38cae6d05d865ul,0x3ca3ba556658c94dul,
    0x3ff3a7db34e59ff6ul,0x3cb12ee4dbdbae86ul,0x3ff3c32dc313a8e4ul,0x3ca345c8212d2278ul,
    0x3ff3dea64c123422ul,0x3c8f31b9fb32e2e4ul,0x3ff3fa4504ac801bul,0x3ca7378cb7144a8aul,
    0x3ff4160a21f72e29ul,0x3cb1d4e28014ce07ul,0x3ff431f5d950a896ul,0x3caf7ec78b0a541aul,
    0x3ff44e086061892dul,0x3c4bf96ee8afb9fbul,0x3ff46a41ed1d0057ul,0x3ca0283b91b051a3ul,
    0x3ff486a2b5c13cd0ul,0x3c7637bd8150db7dul,0x3ff4a32af0d7d3deul,0x3c9cd9f521d33a36ul,
    0x3ff4bfdad5362a27ul,0x3c804715f70da237ul,0x3ff4dcb299fddd0dul,0x3c9b942d586961a7ul,
    0x3ff4f9b2769d2ca6ul,0x3ca7d36cccde36a5ul,0x3ff516daa2cf6641ul,0x3caa6a7e10f26745ul,
    0x3ff5342b569d4f81ul,0x3cae59e8ae6a99deul,0x3ff551a4ca5d920eul,0x3caab0010b956f3ful,
    0x3ff56f4736b527daul,0x3c9bb5c29f573dd8ul,0x3ff58d12d497c7fdul,0x3c83e81d7478cc39ul,
    0x3ff5ab07dd485429ul,0x3c97a560f05eeddbul,0x3ff5c9268a5946b7ul,0x3c3dfa723ae1bf17ul,
    0x3ff5e76f15ad2148ul,0x3c9d2407ce0f6433ul,0x3ff605e1b976dc08ul,0x3ca71eb321c101b5ul,
    0x3ff6247eb03a5584ul,0x3ca72ffea320da87ul,0x3ff6434634ccc31ful,0x3ca9d919a952ca95ul,
    0x3ff6623882552224ul,0x3ca2b620c0998aacul,0x3ff68155d44ca973ul,0x3c60a350f225390eul,
    0x3ff6a09e667f3bccul,0x3ca26e8d1ea7de88ul,0x3ff6c012750bdabeul,0x3cae1e012931967dul,
    0x3ff6dfb23c651a2eul,0x3cae8b028d2e51bbul,0x3ff6ff7df9519483ul,0x3caa098f71bcc9b0ul,
    0x3ff71f75e8ec5f73ul,0x3cab984227e443e3ul,0x3ff73f9a48a58173ul,0x3ca7807bd5e512c0ul,
    0x3ff75feb564267c8ul,0x3ca7a16a40378bdbul,0x3ff780694fde5d3ful,0x3c97f77c42eaf147ul,
    0x3ff7a11473eb0186ul,0x3caa5b034b27b244ul,0x3ff7c1ed0130c132ul,0x3c9e308095c8fab2ul,
    0x3ff7e2f336cf4e62ul,0x3c6fa03bd08bcda3ul,0x3ff80427543e1a11ul,0x3ca5df63c009a845ul,
    0x3ff82589994cce12ul,0x3ca0964d359847c9ul,0x3ff8471a4623c7acul,0x3ca8857d98c4418dul,
    0x3ff868d99b4492ecul,0x3c9e78c3e7b184a0ul,0x3ff88ac7d98a6699ul,0x3c980f72957b0484ul,
    0x3ff8ace5422aa0dbul,0x3c956f6cd720fba3ul,0x3ff8cf3216b5448bul,0x3cabd0cf46f2f2b2ul,
    0x3ff8f1ae99157736ul,0x3c842bc69235780ful,0x3ff9145b0b91ffc5ul,0x3c9f715c3d56cd59ul,
    0x3ff93737b0cdc5e4ul,0x3cabf4d1acf902dbul,0x3ff95a44cbc8520eul,0x3caa991ae1b50822ul,
    0x3ff97d829fde4e4ful,0x3c9f9bcff58b04aaul,0x3ff9a0f170ca07b9ul,0x3ca4f656fe0ffb05ul,
    0x3ff9c49182a3f090ul,0x3c7984806660caa6ul,0x3ff9e86319e32323ul,0x3c7582eb35cac3f3ul,
    0x3ffa0c667b5de564ul,0x3ca3c8ba7b0b1398ul,0x3ffa309bec4a2d33ul,0x3c938e88295250eful,
    0x3ffa5503b23e255cul,0x3c9e84e556f025f4ul,0x3ffa799e1330b358ul,0x3c983bde4707301aul,
    0x3ffa9e6b5579fdbful,0x3c8d72d3f1e40a93ul,0x3ffac36bbfd3f379ul,0x3ca4cacf4c4b05e2ul,
    0x3ffae89f995ad3adul,0x3c9445b6038d7018ul,0x3ffb0e07298db665ul,0x3c9ed34583749cd0ul,
    0x3ffb33a2b84f15faul,0x3caa2c7497c16670ul,0x3ffb59728de55939ul,0x3c9e0244cd9df3b7ul,
    0x3ffb7f76f2fb5e46ul,0x3ca89f65d4b96070ul,0x3ffba5b030a10649ul,0x3c9b9006b1da3a39ul,
    0x3ffbcc1e904bc1d2ul,0x3c7e4bb29d39c3a3ul,0x3ffbf2c25bd71e08ul,0x3c9b4338a23cfc57ul,
    0x3ffc199bdd85529cul,0x3c7c08f260a0042aul,0x3ffc40ab5fffd07aul,0x3c9647d6355e9b74ul,
    0x3ffc67f12e57d14bul,0x3c8e1e92a1c04f0dul,0x3ffc8f6d9406e7b5ul,0x3c6c920234dbf34dul,
    0x3ffcb720dcef9069ul,0x3c70e49a8304ef1ful,0x3ffcdf0b555dc3f9ul,0x3ca39e87e388d84eul,
    0x3ffd072d4a07897bul,0x3c9c0ae7070a6c51ul,0x3ffd2f87080d89f1ul,0x3c9b7531d7fbf731ul,
    0x3ffd5818dcfba487ul,0x3c7dc682d68358e2ul,0x3ffd80e316c98397ul,0x3ca256cea656ebcdul,
    0x3ffda9e603db3285ul,0x3c95e51917091778ul,0x3ffdd321f301b460ul,0x3c8d2ec7bfc90fd9ul,
    0x3ffdfc97337b9b5eul,0x3ca1d72cedf8d91cul,0x3ffe264614f5a128ul,0x3c9ed88162f63e73ul,
    0x3ffe502ee78b3ff6ul,0x3c7de1319abb2ae2ul,0x3ffe7a51fbc74c83ul,0x3c8c86cfb6cd46e4ul,
    0x3ffea4afa2a490d9ul,0x3c9926f89159b0a1ul,0x3ffecf482d8e67f0ul,0x3c9a89ed8a4d4206ul,
    0x3ffefa1bee615a27ul,0x3c963124664aaf80ul,0x3fff252b376bba97ul,0x3c8d19790d732ab4ul,
    0x3fff50765b6e4540ul,0x3c9309e68da0499ful,0x3fff7bfdad9cbe13ul,0x3c991ff9e4e92cf9ul,
    0x3fffa7c1819e90d8ul,0x3c80fa4931906b80ul,0x3fffd3c22b8f71f1ul,0x3c5b71923d729650ul,
'''


_EXP64_SEGMENT_COEFFICIENTS = r'''
    0x40671547652b82feul, 0x3ce777d0ffda0d24ul, 0x408ef1584af17ecdul, 0xc0513b4d4879e55eul, 0x4114ffde7cc02cf4ul, 0x4118b31b87eb8904ul, 0x3d25e52272e0eaecul
'''

# Keep raw rows as text instead of retaining a Python object for every word.
# The segmented exp payload is presented as either two or four words per row.
_EXP64_SEGMENT_ROWS = r'''
    0x3ff0000000000000ul, 0x0000000000000000ul
    0x3ff0163da9fb3335ul, 0x3d13a4bc88ab6f0bul
    0x3ff02c9a3e778060ul, 0x3d25448c4f3a3faaul
    0x3ff04315e86e7f84ul, 0x3d20cea8d139df85ul
    0x3ff059b0d3158574ul, 0x3d04ca556a1b2fa8ul
    0x3ff0706b29ddf6ddul, 0x3d2173b2871f2217ul
    0x3ff0874518759bc8ul, 0x3cd87a10f84f53b7ul
    0x3ff09e3ecac6f383ul, 0x3d0801b99f4f5b18ul
    0x3ff0b5586cf9890ful, 0x3d1106d9ba253401ul
    0x3ff0cc922b7247f7ul, 0x3d06269675ff4e35ul
    0x3ff0e3ec32d3d1a2ul, 0x3cb62d2d983e6844ul
    0x3ff0fb66affed31aul, 0x3d24935522e05558ul
    0x3ff11301d0125b50ul, 0x3d1bdebf5fb67959ul
    0x3ff12abdc06c31cbul, 0x3d25245852cc8960ul
    0x3ff1429aaea92ddful, 0x3d1df748a567f8daul
    0x3ff15a98c8a58e51ul, 0x3cf846f8d141b1d0ul
    0x3ff172b83c7d517aul, 0x3d224349c9dfc205ul
    0x3ff18af9388c8de9ul, 0x3d1ee14f25e66832ul
    0x3ff1a35beb6fcb75ul, 0x3d03dd2c819e0908ul
    0x3ff1bbe084045cd3ul, 0x3d192b6b9f4c306aul
    0x3ff1d4873168b9aaul, 0x3d136c3949dfc34eul
    0x3ff1ed5022fcd91cul, 0x3d1db25aa6565c95ul
    0x3ff2063b88628cd6ul, 0x3d03118ae698d206ul
    0x3ff21f49917ddc96ul, 0x3cf7c551db68456ful
    0x3ff2387a6e756238ul, 0x3d1045c6a6d3eecbul
    0x3ff251ce4fb2a63ful, 0x3d00db2bd14acf32ul
    0x3ff26b4565e27cddul, 0x3cf77bf4b6a48214ul
    0x3ff284dfe1f56380ul, 0x3d177f3dc394bd27ul
    0x3ff29e9df51fdee1ul, 0x3cfb5d8e0885ab1bul
    0x3ff2b87fd0dad98ful, 0x3d2390730108bc38ul
    0x3ff2d285a6e4030bul, 0x3d03a1fee8c9225cul
    0x3ff2ecafa93e2f56ul, 0x3ce5b2c9eff60911ul
    0x3ff306fe0a31b715ul, 0x3cfbd9046b69ea24ul
    0x3ff32170fc4cd831ul, 0x3d000e3e54fb7d75ul
    0x3ff33c08b26416fful, 0x3d06fc3180aac683ul
    0x3ff356c55f929ff0ul, 0x3d1e07c0017dfc4aul
    0x3ff371a7373aa9caul, 0x3d18cb357ddac301ul
    0x3ff38cae6d05d865ul, 0x3d13ba556658c94dul
    0x3ff3a7db34e59ff6ul, 0x3d212ee4dbdbae86ul
    0x3ff3c32dc313a8e4ul, 0x3d1345c8212d2278ul
    0x3ff3dea64c123422ul, 0x3cff31b9fb32e2e4ul
    0x3ff3fa4504ac801bul, 0x3d17378cb7144a8aul
    0x3ff4160a21f72e29ul, 0x3d21d4e28014ce07ul
    0x3ff431f5d950a896ul, 0x3d1f7ec78b0a541aul
    0x3ff44e086061892dul, 0x3cbbf96ee8afb9fbul
    0x3ff46a41ed1d0057ul, 0x3d10283b91b051a3ul
    0x3ff486a2b5c13cd0ul, 0x3ce637bd8150db7dul
    0x3ff4a32af0d7d3deul, 0x3d0cd9f521d33a36ul
    0x3ff4bfdad5362a27ul, 0x3cf04715f70da237ul
    0x3ff4dcb299fddd0dul, 0x3d0b942d586961a7ul
    0x3ff4f9b2769d2ca6ul, 0x3d17d36cccde36a5ul
    0x3ff516daa2cf6641ul, 0x3d1a6a7e10f26745ul
    0x3ff5342b569d4f81ul, 0x3d1e59e8ae6a99deul
    0x3ff551a4ca5d920eul, 0x3d1ab0010b956f3ful
    0x3ff56f4736b527daul, 0x3d0bb5c29f573dd8ul
    0x3ff58d12d497c7fdul, 0x3cf3e81d7478cc39ul
    0x3ff5ab07dd485429ul, 0x3d07a560f05eeddbul
    0x3ff5c9268a5946b7ul, 0x3cadfa723ae1bf17ul
    0x3ff5e76f15ad2148ul, 0x3d0d2407ce0f6433ul
    0x3ff605e1b976dc08ul, 0x3d171eb321c101b5ul
    0x3ff6247eb03a5584ul, 0x3d172ffea320da87ul
    0x3ff6434634ccc31ful, 0x3d19d919a952ca95ul
    0x3ff6623882552224ul, 0x3d12b620c0998aacul
    0x3ff68155d44ca973ul, 0x3cd0a350f225390eul
    0x3ff6a09e667f3bccul, 0x3d126e8d1ea7de88ul
    0x3ff6c012750bdabeul, 0x3d1e1e012931967dul
    0x3ff6dfb23c651a2eul, 0x3d1e8b028d2e51bbul
    0x3ff6ff7df9519483ul, 0x3d1a098f71bcc9b0ul
    0x3ff71f75e8ec5f73ul, 0x3d1b984227e443e3ul
    0x3ff73f9a48a58173ul, 0x3d17807bd5e512c0ul
    0x3ff75feb564267c8ul, 0x3d17a16a40378bdbul
    0x3ff780694fde5d3ful, 0x3d07f77c42eaf147ul
    0x3ff7a11473eb0186ul, 0x3d1a5b034b27b244ul
    0x3ff7c1ed0130c132ul, 0x3d0e308095c8fab2ul
    0x3ff7e2f336cf4e62ul, 0x3cdfa03bd08bcda3ul
    0x3ff80427543e1a11ul, 0x3d15df63c009a845ul
    0x3ff82589994cce12ul, 0x3d10964d359847c9ul
    0x3ff8471a4623c7acul, 0x3d18857d98c4418dul
    0x3ff868d99b4492ecul, 0x3d0e78c3e7b184a0ul
    0x3ff88ac7d98a6699ul, 0x3d080f72957b0484ul
    0x3ff8ace5422aa0dbul, 0x3d056f6cd720fba3ul
    0x3ff8cf3216b5448bul, 0x3d1bd0cf46f2f2b2ul
    0x3ff8f1ae99157736ul, 0x3cf42bc69235780ful
    0x3ff9145b0b91ffc5ul, 0x3d0f715c3d56cd59ul
    0x3ff93737b0cdc5e4ul, 0x3d1bf4d1acf902dbul
    0x3ff95a44cbc8520eul, 0x3d1a991ae1b50822ul
    0x3ff97d829fde4e4ful, 0x3d0f9bcff58b04aaul
    0x3ff9a0f170ca07b9ul, 0x3d14f656fe0ffb05ul
    0x3ff9c49182a3f090ul, 0x3ce984806660caa6ul
    0x3ff9e86319e32323ul, 0x3ce582eb35cac3f3ul
    0x3ffa0c667b5de564ul, 0x3d13c8ba7b0b1398ul
    0x3ffa309bec4a2d33ul, 0x3d038e88295250eful
    0x3ffa5503b23e255cul, 0x3d0e84e556f025f4ul
    0x3ffa799e1330b358ul, 0x3d083bde4707301aul
    0x3ffa9e6b5579fdbful, 0x3cfd72d3f1e40a93ul
    0x3ffac36bbfd3f379ul, 0x3d14cacf4c4b05e2ul
    0x3ffae89f995ad3adul, 0x3d0445b6038d7018ul
    0x3ffb0e07298db665ul, 0x3d0ed34583749cd0ul
    0x3ffb33a2b84f15faul, 0x3d1a2c7497c16670ul
    0x3ffb59728de55939ul, 0x3d0e0244cd9df3b7ul
    0x3ffb7f76f2fb5e46ul, 0x3d189f65d4b96070ul
    0x3ffba5b030a10649ul, 0x3d0b9006b1da3a39ul
    0x3ffbcc1e904bc1d2ul, 0x3cee4bb29d39c3a3ul
    0x3ffbf2c25bd71e08ul, 0x3d0b4338a23cfc57ul
    0x3ffc199bdd85529cul, 0x3cec08f260a0042aul
    0x3ffc40ab5fffd07aul, 0x3d0647d6355e9b74ul
    0x3ffc67f12e57d14bul, 0x3cfe1e92a1c04f0dul
    0x3ffc8f6d9406e7b5ul, 0x3cdc920234dbf34dul
    0x3ffcb720dcef9069ul, 0x3ce0e49a8304ef1ful
    0x3ffcdf0b555dc3f9ul, 0x3d139e87e388d84eul
    0x3ffd072d4a07897bul, 0x3d0c0ae7070a6c51ul
    0x3ffd2f87080d89f1ul, 0x3d0b7531d7fbf731ul
    0x3ffd5818dcfba487ul, 0x3cedc682d68358e2ul
    0x3ffd80e316c98397ul, 0x3d1256cea656ebcdul
    0x3ffda9e603db3285ul, 0x3d05e51917091778ul
    0x3ffdd321f301b460ul, 0x3cfd2ec7bfc90fd9ul
    0x3ffdfc97337b9b5eul, 0x3d11d72cedf8d91cul
    0x3ffe264614f5a128ul, 0x3d0ed88162f63e73ul
    0x3ffe502ee78b3ff6ul, 0x3cede1319abb2ae2ul
    0x3ffe7a51fbc74c83ul, 0x3cfc86cfb6cd46e4ul
    0x3ffea4afa2a490d9ul, 0x3d0926f89159b0a1ul
    0x3ffecf482d8e67f0ul, 0x3d0a89ed8a4d4206ul
    0x3ffefa1bee615a27ul, 0x3d063124664aaf80ul
    0x3fff252b376bba97ul, 0x3cfd19790d732ab4ul
    0x3fff50765b6e4540ul, 0x3d0309e68da0499ful
    0x3fff7bfdad9cbe13ul, 0x3d091ff9e4e92cf9ul
    0x3fffa7c1819e90d8ul, 0x3cf0fa4931906b80ul
    0x3fffd3c22b8f71f1ul, 0x3ccb71923d729650ul
'''

# The fourth word in padded log tables is a literal zero, not another coefficient.
_LOG64_ROWS = r'''
    0x3ff0000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul
    0x3fefc07f00000000ul, 0x3f7fe02b00000000ul, 0x3e1ac419e43f0ddaul
    0x3fef81f800000000ul, 0x3f8fc0b100000000ul, 0xbe23c0fe070c179ful
    0x3fef446600000000ul, 0x3f97b90f00000000ul, 0xbe3e0a8ed755c02ful
    0x3fef07c200000000ul, 0x3f9f829900000000ul, 0x3e0cf0670099f1a3ul
    0x3feecc0800000000ul, 0x3fa39e8300000000ul, 0xbe41804f1815c244ul
    0x3fee913200000000ul, 0x3fa7745300000000ul, 0x3e4d8cb9231189c5ul
    0x3fee573b00000000ul, 0x3fab42da00000000ul, 0xbe3773457c1ebb72ul
    0x3fee1e1e00000000ul, 0x3faf0a3300000000ul, 0xbe3ff74e2ccf4146ul
    0x3fede5d700000000ul, 0x3fb1653600000000ul, 0xbdf5c84ae979dea4ul
    0x3fedae6000000000ul, 0x3fb341dc00000000ul, 0xbe5a79098bdb544aul
    0x3fed77b600000000ul, 0x3fb51b0a00000000ul, 0x3e3f061c61692f7aul
    0x3fed41d400000000ul, 0x3fb6f0d400000000ul, 0xbe5d46a50d1906c4ul
    0x3fed0cb600000000ul, 0x3fb8c34200000000ul, 0xbe239cbaba14cd62ul
    0x3fecd85700000000ul, 0x3fba926900000000ul, 0x3e3a4adde564f46dul
    0x3feca4b300000000ul, 0x3fbc5e5500000000ul, 0xbe50290e2e1a8a7aul
    0x3fec71c700000000ul, 0x3fbe270800000000ul, 0x3e5b8abcd97a7ab7ul
    0x3fec3f8f00000000ul, 0x3fbfec9100000000ul, 0x3e5076faaeca8b94ul
    0x3fec0e0700000000ul, 0x3fc0d77f00000000ul, 0xbe6ccbdc69665a24ul
    0x3febdd2c00000000ul, 0x3fc1b72900000000ul, 0xbe62b424f6f5c1b2ul
    0x3febacf900000000ul, 0x3fc2955300000000ul, 0x3e6607fd51d3017cul
    0x3feb7d6c00000000ul, 0x3fc371fd00000000ul, 0x3e6007a42e0ef3dful
    0x3feb4e8200000000ul, 0x3fc44d2a00000000ul, 0x3e3996fe04cfa0c4ul
    0x3feb203600000000ul, 0x3fc526e700000000ul, 0x3e43a1b5a0a2e63dul
    0x3feaf28700000000ul, 0x3fc5ff2f00000000ul, 0x3e5853cab1e438a4ul
    0x3feac57000000000ul, 0x3fc6d61000000000ul, 0x3e69c67497235528ul
    0x3fea98ef00000000ul, 0x3fc7ab8b00000000ul, 0xbe56f791d6f20a52ul
    0x3fea6d0200000000ul, 0x3fc87f9f00000000ul, 0xbe62b7cd059bf998ul
    0x3fea41a400000000ul, 0x3fc9525b00000000ul, 0x3e4cf456f476415bul
    0x3fea16d400000000ul, 0x3fca23bc00000000ul, 0xbdcd4a98e6c8eefaul
    0x3fe9ec8f00000000ul, 0x3fcaf3c700000000ul, 0x3e5f406219ec6157ul
    0x3fe9c2d100000000ul, 0x3fcbc28800000000ul, 0xbe1e93868cb00b40ul
    0x3fe9999a00000000ul, 0x3fcc8ff600000000ul, 0xbe5c32b0ef29f269ul
    0x3fe970e500000000ul, 0x3fcd5c2100000000ul, 0x3e60d3eee5e6e443ul
    0x3fe948b100000000ul, 0x3fce270700000000ul, 0x3e678abcb9ba7aa2ul
    0x3fe920fb00000000ul, 0x3fcef0af00000000ul, 0x3e60f716d7a48746ul
    0x3fe8f9c200000000ul, 0x3fcfb91600000000ul, 0x3e56af219d46a30bul
    0x3fe8d30200000000ul, 0x3fd0402400000000ul, 0x3e7b2d34bb56b68ful
    0x3fe8acb900000000ul, 0x3fd0a32500000000ul, 0x3e44e721dfbee7faul
    0x3fe886e600000000ul, 0x3fd1058c00000000ul, 0xbe6728da8f33b032ul
    0x3fe8618600000000ul, 0x3fd1675d00000000ul, 0xbe545459d1fc6323ul
    0x3fe83c9700000000ul, 0x3fd1c89a00000000ul, 0x3e35a674bc6f457aul
    0x3fe8181800000000ul, 0x3fd2294200000000ul, 0x3e6de7bcc2d1214dul
    0x3fe7f40600000000ul, 0x3fd2895a00000000ul, 0x3e47bd0d47bd6926ul
    0x3fe7d05f00000000ul, 0x3fd2e8e300000000ul, 0x3e7ab84788a70ba1ul
    0x3fe7ad2200000000ul, 0x3fd347de00000000ul, 0xbe7359e0a9aa6ea6ul
    0x3fe78a4d00000000ul, 0x3fd3a64b00000000ul, 0xbe24b5b3cdc6edd8ul
    0x3fe767dd00000000ul, 0x3fd4043000000000ul, 0x3e6d4354086e85f7ul
    0x3fe745d100000000ul, 0x3fd4618d00000000ul, 0x3e20e30f13e8af2ful
    0x3fe7242800000000ul, 0x3fd4be6100000000ul, 0xbe4511072e494b13ul
    0x3fe702e000000000ul, 0x3fd51aaf00000000ul, 0xbe7e348174bd8c5bul
    0x3fe6e1f700000000ul, 0x3fd5767800000000ul, 0x3e70d1574add28d3ul
    0x3fe6c16c00000000ul, 0x3fd5d1be00000000ul, 0xbe04fec2b5ee4b93ul
    0x3fe6a13d00000000ul, 0x3fd62c8200000000ul, 0x3e7bae720753dba9ul
    0x3fe6816800000000ul, 0x3fd686c800000000ul, 0x3e77a6c533b110b5ul
    0x3fe661ec00000000ul, 0x3fd6e09000000000ul, 0xbe62ea2da539f1bful
    0x3fe642c800000000ul, 0x3fd739d900000000ul, 0xbe42885bf2c630bdul
    0x3fe623fa00000000ul, 0x3fd792a700000000ul, 0xbe7208ad3040f74dul
    0x3fe6058100000000ul, 0x3fd7eaf900000000ul, 0x3e74e0ac89f92ea8ul
    0x3fe5e75c00000000ul, 0x3fd842d100000000ul, 0x3e443d18d2927070ul
    0x3fe5c98800000000ul, 0x3fd89a3400000000ul, 0x3e3b050b6ad69f0dul
    0x3fe5ac0500000000ul, 0x3fd8f12000000000ul, 0xbe6e64cd1630ef47ul
    0x3fe58ed200000000ul, 0x3fd9479500000000ul, 0xbe74f7ba18d16c8cul
    0x3fe571ed00000000ul, 0x3fd99d9600000000ul, 0x3e6a8bf0c3f5d585ul
    0x3fe5555500000000ul, 0xbfd2696100000000ul, 0xbe5134d992783696ul
    0x3fe5390900000000ul, 0xbfd2144500000000ul, 0x3e7bbc5229784364ul
    0x3fe51d0800000000ul, 0xbfd1bf9a00000000ul, 0x3e77296522c89570ul
    0x3fe5015000000000ul, 0xbfd16b5d00000000ul, 0x3e7d14c12b299048ul
    0x3fe4e5e100000000ul, 0xbfd1179000000000ul, 0x3e7b760771887184ul
    0x3fe4cab900000000ul, 0xbfd0c42f00000000ul, 0x3e624f50ab07466eul
    0x3fe4afd700000000ul, 0xbfd0713a00000000ul, 0x3e7decaa928322d9ul
    0x3fe4953a00000000ul, 0xbfd01eaf00000000ul, 0x3e74764e6ae23044ul
    0x3fe47ae100000000ul, 0xbfcf991b00000000ul, 0x3e64d313e5a09b79ul
    0x3fe460cc00000000ul, 0xbfcef5af00000000ul, 0xbe61373f8077bb84ul
    0x3fe446f800000000ul, 0xbfce530c00000000ul, 0xbe6ff9c2748484a3ul
    0x3fe42d6600000000ul, 0xbfcdb13d00000000ul, 0x3e5f95bbd0655f6ful
    0x3fe4141400000000ul, 0xbfcd103700000000ul, 0xbe6c99578ed6061eul
    0x3fe3fb0100000000ul, 0xbfcc6ffa00000000ul, 0xbe577806732e3237ul
    0x3fe3e22d00000000ul, 0xbfcbd08900000000ul, 0x3e47c42a2bf10fedul
    0x3fe3c99600000000ul, 0xbfcb31db00000000ul, 0x3e66290dc716311bul
    0x3fe3b13b00000000ul, 0xbfca93ed00000000ul, 0x3e60dd4997243510ul
    0x3fe3991c00000000ul, 0xbfc9f6c300000000ul, 0x3e48f76adfeca7cdul
    0x3fe3813800000000ul, 0xbfc95a5a00000000ul, 0xbe673dc04fc8a158ul
    0x3fe3698e00000000ul, 0xbfc8beb000000000ul, 0xbe5d9c7f39955aaeul
    0x3fe3521d00000000ul, 0xbfc823c200000000ul, 0x3e6eab971091232dul
    0x3fe33ae400000000ul, 0xbfc7898b00000000ul, 0xbe52a223679ef009ul
    0x3fe323e300000000ul, 0xbfc6f01100000000ul, 0x3e6922a64158e0c6ul
    0x3fe30d1900000000ul, 0xbfc6574f00000000ul, 0x3e625cfb3193a69eul
    0x3fe2f68500000000ul, 0xbfc5bf4200000000ul, 0xbe55aa1d50fdc4dcul
    0x3fe2e02600000000ul, 0xbfc527e800000000ul, 0x3e6ad7935301700ful
    0x3fe2c9fb00000000ul, 0xbfc4913b00000000ul, 0xbe6cccec47f79263ul
    0x3fe2b40500000000ul, 0xbfc3fb4800000000ul, 0x3e51336e105b0885ul
    0x3fe29e4100000000ul, 0xbfc365fc00000000ul, 0x3e6bfa9c4b741671ul
    0x3fe288b000000000ul, 0xbfc2d16100000000ul, 0x3e6cde5fc18a4cddul
    0x3fe2735100000000ul, 0xbfc23d7300000000ul, 0xbe4a49be40a47badul
    0x3fe25e2200000000ul, 0xbfc1aa2800000000ul, 0xbe6b88fb7238dab1ul
    0x3fe2492500000000ul, 0xbfc1179200000000ul, 0x3e6f760921086a1dul
    0x3fe2345600000000ul, 0xbfc0859500000000ul, 0xbe69678bc3621ce2ul
    0x3fe21fb800000000ul, 0xbfbfe89800000000ul, 0xbe4cedde72caa542ul
    0x3fe20b4700000000ul, 0xbfbec73900000000ul, 0x3e467af78e5a2c07ul
    0x3fe1f70400000000ul, 0xbfbda72000000000ul, 0xbe58e10b8893e565ul
    0x3fe1e2ef00000000ul, 0xbfbc885500000000ul, 0x3e5390ee9645ca3cul
    0x3fe1cf0700000000ul, 0xbfbb6acd00000000ul, 0xbe46d6a834f00afeul
    0x3fe1bb4a00000000ul, 0xbfba4e7300000000ul, 0x3e57d3929695b9c2ul
    0x3fe1a7b900000000ul, 0xbfb9335900000000ul, 0x3e415362cba90d0bul
    0x3fe1945400000000ul, 0xbfb8198500000000ul, 0xbe5bd0326a07078dul
    0x3fe1811800000000ul, 0xbfb700d200000000ul, 0xbe25d580c1e8d9eful
    0x3fe16e0700000000ul, 0xbfb5e96100000000ul, 0xbe3d977a96fd4b35ul
    0x3fe15b1e00000000ul, 0xbfb4d30c00000000ul, 0x3e416fc839d147d2ul
    0x3fe1485f00000000ul, 0xbfb3bdf500000000ul, 0x3e441708f82856a0ul
    0x3fe135c800000000ul, 0xbfb2aa0400000000ul, 0x3e56ee3a36dd15e9ul
    0x3fe1235900000000ul, 0xbfb1973d00000000ul, 0xbe5051951794557eul
    0x3fe1111100000000ul, 0xbfb0859800000000ul, 0x3e5298719e5dd716ul
    0x3fe0fef000000000ul, 0xbfaeea3000000000ul, 0x3e3ffca44227ca4eul
    0x3fe0ecf500000000ul, 0xbfaccb6700000000ul, 0xbe1bbb144f0c5eb6ul
    0x3fe0db2100000000ul, 0xbfaaaefd00000000ul, 0xbe387d7b098d4926ul
    0x3fe0c97100000000ul, 0xbfa894a100000000ul, 0x3e4ad818d332cbe8ul
    0x3fe0b7e700000000ul, 0xbfa67c9700000000ul, 0xbe44b52e7bd04186ul
    0x3fe0a68100000000ul, 0xbfa466ae00000000ul, 0x3e4af4871e9cdd7aul
    0x3fe0953f00000000ul, 0xbfa252ec00000000ul, 0xbe43e3431b66b143ul
    0x3fe0842100000000ul, 0xbfa0415d00000000ul, 0x3e4d862efee3fa36ul
    0x3fe0732600000000ul, 0xbf9c63d000000000ul, 0xbe3b052a8a631fbbul
    0x3fe0624e00000000ul, 0xbf98493000000000ul, 0xbe24645dcf45db83ul
    0x3fe0519800000000ul, 0xbf9432ab00000000ul, 0xbe22cc06204e64b7ul
    0x3fe0410400000000ul, 0xbf90205200000000ul, 0xbe3624d591d27c39ul
    0x3fe0309200000000ul, 0xbf88246e00000000ul, 0x3e271decb9779881ul
    0x3fe0204100000000ul, 0xbf80105400000000ul, 0xbe2d61f9984b78c6ul
    0x3fe0101000000000ul, 0xbf7007f500000000ul, 0xbe165620cd5f950eul
    0x3fe0000000000000ul, 0x0000000000000000ul, 0x0000000000000000ul
'''


def _ulong_table(
    name: str, size: int, rows: list[str], *,
    separator: str, trailing_comma: bool,
) -> str:
    declaration = f"constant ulong {name}[{size}]={{"
    if not rows:
        return declaration + "\n};"
    body = ",\n".join(rows).replace(", ", separator)
    if trailing_comma:
        body += ","
    return declaration + "\n" + body + "\n};"


def exponential32_table(name: str) -> str:
    """Retain the caller's declaration name and the complete 128-entry payload."""
    return f"constant ulong {name}[128]={{" + _EXP32_VALUES + "};"


def exponential64_tables(coefficients: str, table: str) -> str:
    """Retain both declaration names and their original separator."""
    return (f"constant ulong {coefficients}[5]={{" + _EXP64_COEFFICIENTS
            + f"}};\nconstant ulong {table}[256]={{" + _EXP64_VALUES + "};")


def exponential64_segment_table(name: str, *, four_columns: bool = False) -> str:
    """Keep each provider's two-word or four-word presentation of the same data."""
    rows = _EXP64_SEGMENT_ROWS.splitlines()[1:]
    if four_columns:
        rows = [rows[index] + ", " + rows[index + 1].lstrip()
                for index in range(0, len(rows), 2)]
    return _ulong_table(name, 256, rows, separator="," if four_columns else ", ",
                        trailing_comma=four_columns)


def exponential64_segment_tables(coefficients: str, table: str) -> str:
    """Retain the distinct seven-coefficient segmented exponential payload."""
    return (f"constant ulong {coefficients}[7]={{" + _EXP64_SEGMENT_COEFFICIENTS
            + "};\n" + exponential64_segment_table(table))


def logarithm64_table(
    name: str, *, padded: bool, separator: str, trailing_comma: bool,
) -> str:
    """Retain each log provider's declaration size, zero padding and punctuation."""
    rows = _LOG64_ROWS.splitlines()[1:]
    if padded:
        rows = [row + ", 0x0000000000000000ul" for row in rows]
    return _ulong_table(name, 516 if padded else 387, rows,
                        separator=separator, trailing_comma=trailing_comma)
