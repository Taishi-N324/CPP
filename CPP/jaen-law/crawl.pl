#!/usr/bin/perl

`mkdir -p download txt`;

# foreach my $i (10 .. 2235) {
#     if(not -e "download/$i.xml") {
#         print "xml $i\n";
#         `wget "http://www.japaneselawtranslation.go.jp/law/detail_download/?id=$i&ff=01" -O 'download/$i.xml'`;
#         sleep 2;
#     }
#     if(not -e "download/$i.txt") {
#         print "txt $i\n";
#         `wget "http://www.japaneselawtranslation.go.jp/law/detail_download/?id=$i&ff=13" -O 'download/$i.txt'`;
#         sleep 2;
#     }
#     if((not -e "download/$i.ja") and (-e "download/$i.txt")) {
#         print "splitting $i\n";
#         `./split-text.pl download/$i.ja download/$i.en < download/$i.txt`;
#     }
# }

`cat download/*.en > txt/law-corpus.en`;
`cat download/*.ja > txt/law-corpus.ja`;
