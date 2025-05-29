#include <bits/stdc++.h>
using namespace std;

class BPE {
private:
    unordered_map<string, int> str2tok;
    unordered_map<int, string> tok2str;
    map<pair<int, int>, int> merged_token;
public:
    BPE() {}

    vector<string> splitText(const string &text) {
        vector<string> tokens;
        for (int i = 0; i < text.size(); i++) {
            if (text[i] == ' ' && i != 0) tokens.push_back("Ġ");
            if (text[i] != ' ') tokens.push_back(string(1, text[i]));
        }
        return tokens;
    }

    bool isPresent(const vector<string>& target, const string &key) {
        return find(target.begin(), target.end(), key) != target.end();
    }

    pair<pair<int, int>, int> findFreqPair(const vector<int>& token_ids) {
        map<pair<int, int>, int> freq;
        for (int i = 0; i < token_ids.size() - 1; i++) {
            pair<int, int> p = {token_ids[i], token_ids[i+1]};
            freq[p]++;
        }
        pair<int, int> bestPair;
        int bestCount = 0;
        for (auto &pr : freq) {
            if (pr.second > bestCount) {
                bestCount = pr.second;
                bestPair = pr.first;
            }
        }
        return {bestPair, bestCount};
    }

    vector<int> replacePair(const vector<int>& token_ids, const pair<int,int>& pair_id, int new_id) {
        vector<int> result;
        int i = 0;
        while (i < token_ids.size()) {
            if (i < token_ids.size()-1 && token_ids[i] == pair_id.first && token_ids[i+1] == pair_id.second) {
                result.push_back(new_id);
                i += 2;
            } else {
                result.push_back(token_ids[i]);
                i += 1;
            }
        }
        return result;
    }

    void train(const string &text, int vocab_size) {
        vector<string> processed_text = splitText(text);
        vector<string> unique_characters;
        for (int i = 0; i < 256; i++) {
            string s(1, char(i));
            unique_characters.push_back(s);
        }
        set<string> sset(processed_text.begin(), processed_text.end());
        for (auto &ch : sset) {
            if (!isPresent(unique_characters, ch)) unique_characters.push_back(ch);
        }
        for (int i = 0; i < unique_characters.size(); i++) {
            tok2str[i] = unique_characters[i];
            str2tok[unique_characters[i]] = i;
        }
        vector<int> token_ids;
        for (auto &tok : processed_text) {
            token_ids.push_back(str2tok[tok]);
        }
        int next_id = tok2str.size();
        while (next_id < vocab_size) {
            auto best = findFreqPair(token_ids);
            if (best.second <= 0) break;
            pair<int,int> bestPair = best.first;
            token_ids = replacePair(token_ids, bestPair, next_id);
            merged_token[bestPair] = next_id;
            string merged_str = tok2str[bestPair.first] + tok2str[bestPair.second];
            tok2str[next_id] = merged_str;
            str2tok[merged_str] = next_id;
            next_id++;
        }
    }

    vector<string> splitBySpace(const string &text) {
        vector<string> words;
        istringstream iss(text);
        string word;
        while(iss >> word) {
            words.push_back(word);
        }
        return words;
    }

    vector<int> tokenizeWithBPE(const string &token) {
        vector<int> token_ids;
        for (char c : token) {
            string s(1, c);
            if (str2tok.find(s) == str2tok.end()) {
                throw runtime_error("Character not found in vocab: " + s);
            }
            token_ids.push_back(str2tok[s]);
        }
        bool canMerge = true;
        while(canMerge && token_ids.size() > 1) {
            canMerge = false;
            vector<int> new_tokens;
            int i = 0;
            while(i < token_ids.size()) {
                if(i < token_ids.size()-1) {
                    pair<int,int> pr = {token_ids[i], token_ids[i+1]};
                    if(merged_token.find(pr) != merged_token.end()) {
                        new_tokens.push_back(merged_token[pr]);
                        i += 2;
                        canMerge = true;
                        continue;
                    }
                }
                new_tokens.push_back(token_ids[i]);
                i++;
            }
            token_ids = new_tokens;
        }
        return token_ids;
    }

    vector<int> encode(const string &text) {
        vector<string> words = splitBySpace(text);
        vector<string> tokens;
        for (int i = 0; i < words.size(); i++) {
            if(i > 0) tokens.push_back("Ġ" + words[i]);
            else tokens.push_back(words[i]);
        }
        vector<int> token_ids;
        for (auto &token : tokens) {
            if (str2tok.find(token) != str2tok.end()) {
                token_ids.push_back(str2tok[token]);
            } else {
                vector<int> sub_tokens = tokenizeWithBPE(token);
                token_ids.insert(token_ids.end(), sub_tokens.begin(), sub_tokens.end());
            }
        }
        return token_ids;
    }

    string decode(const vector<int> &token_ids) {
        string decoded;
        for (auto id : token_ids) {
            if (tok2str.find(id) == tok2str.end()) throw runtime_error("Token ID not found in vocab");
            string token = tok2str[id];
            if(token.size() >= 2 && token.substr(0, 2) == "Ġ") {
                decoded += " " + token.substr(2);
            } else if(token.size() >= 1 && token[0] == 'Ġ') {
                decoded += " " + token.substr(1);
            } else {
                decoded += token;
            }
        }
        return decoded;
    }
};

vector<int> returnByteBuffer(const string &s) {
    vector<int> byteBuffer;
    for (char c : s) {
        byteBuffer.push_back((int)c);
    }
    return byteBuffer;
}

void printList(const vector<int> &toPrint) {
    for (auto num : toPrint) {
        cout << num << " ";
    }
    cout << endl;
}

int main(){
    printList(returnByteBuffer("This is some text"));
    BPE bpe;
    string sample = "This is some text";
    bpe.train(sample, 300);
    vector<int> encoded = bpe.encode(sample);
    printList(encoded);
    string decoded = bpe.decode(encoded);
    cout << decoded << endl;
    return 0;
}
